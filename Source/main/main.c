#include <stdbool.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_err.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_event.h"
#include "esp_app_desc.h"
#include "esp_idf_version.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

#include "lwip/inet.h"
#include "lwip/sockets.h"

#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "host/ble_uuid.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#include "vl53l8cx_api.h"
#include "esp_timer.h"
#include "rps_classifier.h"
#include "rps_telemetry.h"
#include "fall_classifier.h"
#include "fall_telemetry.h"

#if defined(CONFIG_FALLINGDETECT_FALL_ENABLED) && CONFIG_FALLINGDETECT_RANGING_FREQUENCY_HZ != 10
#error "The fall LSTM was trained at 10 Hz; retrain before changing the sensor frequency"
#endif

#define TAG "falling_detect"
#define SENSOR_I2C_PORT I2C_NUM_0
#define SENSOR_I2C_ADDRESS (VL53L8CX_DEFAULT_I2C_ADDRESS >> 1)
#define SENSOR_ZONE_COUNT VL53L8CX_RESOLUTION_8X8
#define SENSOR_INT_GPIO GPIO_NUM_20
#define BEEP_GPIO GPIO_NUM_40
#define GREEN_LED_GPIO GPIO_NUM_5
#define RED_LED_GPIO GPIO_NUM_6
#define WIFI_UDP_PORT 3333
#define WIFI_SSID_MAX_LEN 32
#define WIFI_PASSWORD_MAX_LEN 64

/* Custom BLE service: one 20-byte gesture notification per sensor frame.
 * Distance frames continue to use Wi-Fi UDP. */
static const ble_uuid128_t telemetry_service_uuid =
    BLE_UUID128_INIT(0xF0, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
                     0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12);
static const ble_uuid128_t telemetry_characteristic_uuid =
    BLE_UUID128_INIT(0xF1, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
                     0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12);
static uint16_t telemetry_value_handle;
static volatile uint16_t telemetry_conn_handle = BLE_HS_CONN_HANDLE_NONE;
static volatile bool telemetry_notify_enabled;
static uint8_t telemetry_addr_type;

/* BLE provisioning characteristic. The phone writes:
 * 0x01, ssid_len, password_len, ssid bytes, password bytes. */
static const ble_uuid128_t provisioning_characteristic_uuid =
    BLE_UUID128_INIT(0xF3, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
                     0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12);
static const ble_uuid128_t firmware_info_uuid =
    BLE_UUID128_INIT(0xF2, 0xDE, 0xBC, 0x9A, 0x78, 0x56, 0x34, 0x12,
                     0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12);
static uint16_t provisioning_value_handle;
static uint16_t firmware_info_value_handle;
static int wifi_udp_socket = -1;
static struct sockaddr_in wifi_broadcast_address;
static TaskHandle_t wifi_config_task_handle;
static char configured_ssid[WIFI_SSID_MAX_LEN + 1];
static char configured_password[WIFI_PASSWORD_MAX_LEN + 1];

static bool is_valid_measurement(uint8_t status);

static int firmware_info_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                                   struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)arg;
    if (ctxt->op != BLE_GATT_ACCESS_OP_READ_CHR) {
        return BLE_ATT_ERR_READ_NOT_PERMITTED;
    }
    const esp_app_desc_t *app = esp_app_get_description();
    char info[160];
    int len = snprintf(info, sizeof(info),
                       "project=falling_detect;version=%s;idf=%s;chip=ESP32-S3;fall_lstm=1;fall_packet=A8v1",
                       app->version, IDF_VER);
    return os_mbuf_append(ctxt->om, info, (size_t)len) == 0
               ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
}

static int provisioning_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                                  struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)arg;
    if (ctxt->op != BLE_GATT_ACCESS_OP_WRITE_CHR || ctxt->om == NULL) {
        return BLE_ATT_ERR_UNLIKELY;
    }
    uint8_t payload[1 + 1 + 1 + WIFI_SSID_MAX_LEN + WIFI_PASSWORD_MAX_LEN] = {0};
    uint16_t length = OS_MBUF_PKTLEN(ctxt->om);
    if (length > sizeof(payload) || ble_hs_mbuf_to_flat(ctxt->om, payload, sizeof(payload), NULL) != 0 ||
        length < 3 || payload[0] != 0x01 || payload[1] == 0 || payload[1] > WIFI_SSID_MAX_LEN ||
        payload[2] > WIFI_PASSWORD_MAX_LEN || length != (uint16_t)(3 + payload[1] + payload[2])) {
        return BLE_ATT_ERR_INVALID_ATTR_VALUE_LEN;
    }
    memset(configured_ssid, 0, sizeof(configured_ssid));
    memset(configured_password, 0, sizeof(configured_password));
    memcpy(configured_ssid, &payload[3], payload[1]);
    memcpy(configured_password, &payload[3 + payload[1]], payload[2]);
    nvs_handle_t nvs;
    if (nvs_open("wifi", NVS_READWRITE, &nvs) == ESP_OK) {
        nvs_set_str(nvs, "ssid", configured_ssid);
        nvs_set_str(nvs, "password", configured_password);
        nvs_commit(nvs);
        nvs_close(nvs);
    }
    ESP_LOGI(TAG, "Wi-Fi credentials received for SSID '%s'", configured_ssid);
    if (wifi_config_task_handle != NULL) {
        xTaskNotifyGive(wifi_config_task_handle);
    }
    return 0;
}

static int telemetry_access_cb(uint16_t conn_handle, uint16_t attr_handle,
                               struct ble_gatt_access_ctxt *ctxt, void *arg)
{
    (void)conn_handle;
    (void)attr_handle;
    (void)arg;
    return ctxt->op == BLE_GATT_ACCESS_OP_READ_CHR ? 0 : BLE_ATT_ERR_UNLIKELY;
}

static const struct ble_gatt_svc_def telemetry_services[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &telemetry_service_uuid.u,
        .characteristics = (struct ble_gatt_chr_def[]) {
            {
                .uuid = &firmware_info_uuid.u,
                .access_cb = firmware_info_access_cb,
                .flags = BLE_GATT_CHR_F_READ,
                .val_handle = &firmware_info_value_handle,
            },
            {
                .uuid = &telemetry_characteristic_uuid.u,
                .access_cb = telemetry_access_cb,
                .flags = BLE_GATT_CHR_F_NOTIFY,
                .val_handle = &telemetry_value_handle,
            },
            {
                .uuid = &provisioning_characteristic_uuid.u,
                .access_cb = provisioning_access_cb,
                .flags = BLE_GATT_CHR_F_WRITE,
                .val_handle = &provisioning_value_handle,
            },
            {0},
        },
    },
    {0},
};

static void ble_start_advertising(void);

static int ble_gap_event(struct ble_gap_event *event, void *arg)
{
    (void)arg;
    switch (event->type) {
    case BLE_GAP_EVENT_CONNECT:
        if (event->connect.status == 0) {
            telemetry_conn_handle = event->connect.conn_handle;
            telemetry_notify_enabled = false;
            ESP_LOGI(TAG, "BLE connected, handle=%u", telemetry_conn_handle);
        } else {
            ble_start_advertising();
        }
        return 0;
    case BLE_GAP_EVENT_DISCONNECT:
        telemetry_conn_handle = BLE_HS_CONN_HANDLE_NONE;
        telemetry_notify_enabled = false;
        ESP_LOGI(TAG, "BLE disconnected, reason=%d", event->disconnect.reason);
        ble_start_advertising();
        return 0;
    case BLE_GAP_EVENT_ADV_COMPLETE:
        ble_start_advertising();
        return 0;
    case BLE_GAP_EVENT_SUBSCRIBE:
        if (event->subscribe.attr_handle == telemetry_value_handle) {
            telemetry_conn_handle = event->subscribe.conn_handle;
            telemetry_notify_enabled = event->subscribe.cur_notify;
            ESP_LOGI(TAG, "BLE telemetry notifications %s",
                     telemetry_notify_enabled ? "enabled" : "disabled");
        }
        return 0;
    default:
        return 0;
    }
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    (void)arg;
    (void)event_data;
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (wifi_udp_socket >= 0) {
            close(wifi_udp_socket);
            wifi_udp_socket = -1;
        }
        esp_wifi_connect();
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        wifi_udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        if (wifi_udp_socket >= 0) {
            int enable = 1;
            setsockopt(wifi_udp_socket, SOL_SOCKET, SO_BROADCAST, &enable, sizeof(enable));
            memset(&wifi_broadcast_address, 0, sizeof(wifi_broadcast_address));
            wifi_broadcast_address.sin_family = AF_INET;
            wifi_broadcast_address.sin_port = htons(WIFI_UDP_PORT);
            wifi_broadcast_address.sin_addr.s_addr = htonl(INADDR_BROADCAST);
            ESP_LOGI(TAG, "Wi-Fi connected; UDP telemetry port %d", WIFI_UDP_PORT);
        }
    }
}

static void wifi_config_task(void *arg)
{
    (void)arg;
    wifi_config_task_handle = xTaskGetCurrentTaskHandle();
    nvs_handle_t nvs;
    size_t ssid_length = sizeof(configured_ssid);
    size_t password_length = sizeof(configured_password);
    if (nvs_open("wifi", NVS_READONLY, &nvs) == ESP_OK) {
        nvs_get_str(nvs, "ssid", configured_ssid, &ssid_length);
        nvs_get_str(nvs, "password", configured_password, &password_length);
        nvs_close(nvs);
    }
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t init_config = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init_config));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL));
    wifi_config_t wifi_config = {0};
    if (configured_ssid[0] != '\0') {
        strlcpy((char *)wifi_config.sta.ssid, configured_ssid, sizeof(wifi_config.sta.ssid));
        strlcpy((char *)wifi_config.sta.password, configured_password, sizeof(wifi_config.sta.password));
    }
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        strlcpy((char *)wifi_config.sta.ssid, configured_ssid, sizeof(wifi_config.sta.ssid));
        strlcpy((char *)wifi_config.sta.password, configured_password, sizeof(wifi_config.sta.password));
        esp_wifi_disconnect();
        esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
        esp_wifi_connect();
    }
}

static void ble_start_advertising(void)
{
    struct ble_hs_adv_fields fields = {0};
    struct ble_gap_adv_params params = {0};
    const char *name = ble_svc_gap_device_name();
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)name;
    fields.name_len = strlen(name);
    fields.name_is_complete = 1;
    int rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "BLE advertising data setup failed: %d", rc);
        return;
    }
    params.conn_mode = BLE_GAP_CONN_MODE_UND;
    params.disc_mode = BLE_GAP_DISC_MODE_GEN;
    rc = ble_gap_adv_start(telemetry_addr_type, NULL, BLE_HS_FOREVER,
                           &params, ble_gap_event, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "BLE advertising start failed: %d", rc);
        return;
    }
    ESP_LOGI(TAG, "BLE advertising as %s", name);
}

static void ble_on_sync(void)
{
    int rc = ble_hs_id_infer_auto(0, &telemetry_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "BLE address setup failed: %d", rc);
        return;
    }
    ble_start_advertising();
}

static void ble_on_reset(int reason)
{
    ESP_LOGE(TAG, "BLE host reset: %d", reason);
}

static void ble_host_task(void *arg)
{
    (void)arg;
    nimble_port_run();
    vTaskDelete(NULL);
}

static esp_err_t ble_init(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        return err;
    }
    err = nimble_port_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "NimBLE init failed: %s", esp_err_to_name(err));
        return err;
    }
    ble_svc_gap_init();
    if (ble_svc_gap_device_name_set("FallingDetect") != 0) {
        return ESP_FAIL;
    }
    ble_svc_gatt_init();
    int rc = ble_gatts_count_cfg(telemetry_services);
    if (rc != 0) {
        return ESP_FAIL;
    }
    rc = ble_gatts_add_svcs(telemetry_services);
    if (rc != 0) {
        return ESP_FAIL;
    }
    ble_hs_cfg.reset_cb = ble_on_reset;
    ble_hs_cfg.sync_cb = ble_on_sync;
    xTaskCreate(ble_host_task, "ble_host", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "BLE telemetry ready (subscribe to characteristic notifications)");
    return ESP_OK;
}

#ifdef CONFIG_FALLINGDETECT_RPS_ENABLED
static void ble_send_gesture(uint16_t frame, const rps_result_t *prediction,
                              rps_label_t stable)
{
    const uint16_t conn = telemetry_conn_handle;
    if (!telemetry_notify_enabled || conn == BLE_HS_CONN_HANDLE_NONE) return;
    uint8_t payload[RPS_TELEMETRY_BYTES];
    rps_telemetry_encode(payload, frame, prediction, stable);
    struct os_mbuf *om = ble_hs_mbuf_from_flat(payload, sizeof(payload));
    if (om == NULL) {
        ESP_LOGW(TAG, "BLE gesture notification allocation failed");
        return;
    }
    // NimBLE consumes om even when the notification fails.
    int rc = ble_gatts_notify_custom(conn, telemetry_value_handle, om);
    if (rc != 0) ESP_LOGW(TAG, "BLE gesture notification failed: %d", rc);
}
#endif

#ifdef CONFIG_FALLINGDETECT_FALL_ENABLED
#define FALL_FRAME_QUEUE_LENGTH 4
typedef struct {
    int16_t distances[64];
    uint32_t timestamp_ms;
    uint16_t frame_number;
#ifdef FALL_SENSOR_DIAGNOSTICS
    bool diagnostic;
    int16_t raw_distances[64];
    uint8_t target_status[64], target_count[64];
    uint16_t sigma_mm[64];
    uint32_t signal[64], ambient[64];
#endif
} fall_frame_t;

static fall_state_t fall_state;
static bool fall_ready;
static QueueHandle_t fall_frame_queue;
static uint32_t fall_queue_drops;

#ifdef FALL_SENSOR_DIAGNOSTICS
/* Emitted on CPU1, after inference. Acquisition never prints these arrays.
 * Stop automatically after 60 snapshots; this is a bounded diagnostic build. */
static void print_sensor_diagnostic(const fall_frame_t *frame)
{
    if(!frame->diagnostic)return;
    printf("sensor_diag {\"frame\":%u,\"capture_ms\":%" PRIu32 ",\"raw_mm\":[",(unsigned)frame->frame_number,frame->timestamp_ms);
    for(unsigned i=0;i<64;++i)printf("%s%d",i?",":"",frame->raw_distances[i]);
    printf("],\"status\":[");
    for(unsigned i=0;i<64;++i)printf("%s%u",i?",":"",(unsigned)frame->target_status[i]);
    printf("],\"targets\":[");
    for(unsigned i=0;i<64;++i)printf("%s%u",i?",":"",(unsigned)frame->target_count[i]);
    printf("],\"sigma_mm\":[");
    for(unsigned i=0;i<64;++i)printf("%s%u",i?",":"",(unsigned)frame->sigma_mm[i]);
    printf("],\"signal_kcps_spad\":[");
    for(unsigned i=0;i<64;++i)printf("%s%" PRIu32,i?",":"",frame->signal[i]);
    printf("],\"ambient_kcps_spad\":[");
    for(unsigned i=0;i<64;++i)printf("%s%" PRIu32,i?",":"",frame->ambient[i]);
    printf("]}\n");
}
#endif

static void recognize_fall(const fall_frame_t *frame)
{
    if (!fall_ready) return;
    const int64_t start = esp_timer_get_time();
    fall_result_t result;
    fall_push(&fall_state, frame->distances, frame->timestamp_ms, &result);
    result.inference_us = (uint32_t)(esp_timer_get_time() - start);
    const uint32_t queue_ms=(uint32_t)(start/1000)-frame->timestamp_ms;
    uint8_t payload[FALL_TELEMETRY_BYTES];
    fall_telemetry_encode(payload, frame->frame_number, &result);
    const uint16_t conn = telemetry_conn_handle;
    if (telemetry_notify_enabled && conn != BLE_HS_CONN_HANDLE_NONE) {
        struct os_mbuf *om = ble_hs_mbuf_from_flat(payload, sizeof(payload));
        int rc = om == NULL ? BLE_HS_ENOMEM : ble_gatts_notify_custom(conn, telemetry_value_handle, om);
        if (rc != 0) ESP_LOGW(TAG, "BLE fall notification failed: %d", rc);
    }
    if (wifi_udp_socket >= 0) {
        int sent = sendto(wifi_udp_socket, payload, sizeof(payload), MSG_DONTWAIT,
                         (struct sockaddr *)&wifi_broadcast_address, sizeof(wifi_broadcast_address));
        if (sent != sizeof(payload)) ESP_LOGW(TAG, "UDP fall packet send failed");
    }
    gpio_set_level(RED_LED_GPIO, result.stable == FALL_DETECTED || result.stable == FALL_SUSPECTED ? 0 : 1);
    printf("activity frame=%u raw=%d stable=%d p_motion=%.5f p_fall=%.5f p_still=%.5f valid=%u buffered=%u inference_us=%" PRIu32
           " capture_ms=%" PRIu32 " queue_ms=%" PRIu32 " queued=%u queue_drops=%" PRIu32 " updated=%u\n",
           (unsigned)frame->frame_number, (int)result.raw, (int)result.stable,
           (double)result.probability[0], (double)result.probability[1], (double)result.probability[2], (unsigned)result.valid_zones,
           (unsigned)result.buffered_frames, result.inference_us,frame->timestamp_ms,queue_ms,
           (unsigned)uxQueueMessagesWaiting(fall_frame_queue),__atomic_load_n(&fall_queue_drops,__ATOMIC_RELAXED),
           (unsigned)result.prediction_updated);
#ifdef FALL_SENSOR_DIAGNOSTICS
    print_sensor_diagnostic(frame);
#endif
}

static void fall_inference_task(void *arg)
{
    (void)arg;
    fall_frame_t frame;
    while (xQueueReceive(fall_frame_queue, &frame, portMAX_DELAY) == pdTRUE)
        recognize_fall(&frame);
}

static void enqueue_fall_frame(const VL53L8CX_Configuration *device,
                               const VL53L8CX_ResultsData *results, uint32_t now_ms)
{
    fall_frame_t frame = {.timestamp_ms = now_ms, .frame_number = (uint16_t)device->streamcount};
#ifdef FALL_SENSOR_DIAGNOSTICS
    static uint32_t last_diagnostic_ms;
    static unsigned diagnostic_count;
    frame.diagnostic=diagnostic_count<60&&(uint32_t)(now_ms-last_diagnostic_ms)>=1000;
    if(frame.diagnostic){last_diagnostic_ms=now_ms;++diagnostic_count;}
#endif
    for (unsigned zone = 0; zone < 64; ++zone) {
        const unsigned index = VL53L8CX_NB_TARGET_PER_ZONE * zone;
        frame.distances[zone] = is_valid_measurement(results->target_status[index]) && results->distance_mm[index] >= 0
                                  ? results->distance_mm[index] : -1;
#ifdef FALL_SENSOR_DIAGNOSTICS
        if(frame.diagnostic){
            frame.raw_distances[zone]=results->distance_mm[index];
            frame.target_status[zone]=results->target_status[index];
            frame.target_count[zone]=results->nb_target_detected[zone];
            frame.sigma_mm[zone]=results->range_sigma_mm[index];
            frame.signal[zone]=results->signal_per_spad[index];
            frame.ambient[zone]=results->ambient_per_spad[zone];
        }
#endif
    }
    if (xQueueSend(fall_frame_queue, &frame, 0) != pdTRUE) {
        fall_frame_t stale;
        (void)xQueueReceive(fall_frame_queue, &stale, 0);
        (void)xQueueSend(fall_frame_queue, &frame, 0);
        __atomic_add_fetch(&fall_queue_drops,1,__ATOMIC_RELAXED);
        ESP_LOGW(TAG, "fall inference queue overrun; discarded stale frame");
    }
}
#endif

static void wifi_send_frame(const VL53L8CX_Configuration *device,
                            const VL53L8CX_ResultsData *results)
{
    if (wifi_udp_socket < 0) {
        return;
    }
    uint8_t payload[4 + 64 * 2] = {0xA6, (uint8_t)device->streamcount,
                                   (uint8_t)(device->streamcount >> 8), 0x40};
    for (uint8_t zone = 0; zone < 64; ++zone) {
        uint8_t index = (uint8_t)(VL53L8CX_NB_TARGET_PER_ZONE * zone);
        int16_t distance = results->distance_mm[index];
        uint16_t value = is_valid_measurement(results->target_status[index]) && distance >= 0
                             ? (uint16_t)distance : 0xFFFFU;
        payload[4 + zone * 2] = (uint8_t)value;
        payload[5 + zone * 2] = (uint8_t)(value >> 8);
    }
    sendto(wifi_udp_socket, payload, sizeof(payload), MSG_DONTWAIT,
           (struct sockaddr *)&wifi_broadcast_address, sizeof(wifi_broadcast_address));
}

static void indicators_off(void)
{
    const gpio_config_t indicator_config = {
        .pin_bit_mask = (1ULL << GREEN_LED_GPIO) | (1ULL << RED_LED_GPIO),
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&indicator_config));
    /* Board LEDs are active low: high means off. */
    ESP_ERROR_CHECK(gpio_set_level(GREEN_LED_GPIO, 1));
    ESP_ERROR_CHECK(gpio_set_level(RED_LED_GPIO, 1));
}

/* A one-second heartbeat is active only while fresh sensor frames arrive.
 * GPIO5 is active low. No delays are added to the inference loop. */
static void update_running_led(int64_t last_frame_us)
{
    int64_t now = esp_timer_get_time();
    bool receiving = last_frame_us > 0 && now - last_frame_us < 1000000;
    bool on = receiving && (now % 1000000) < 500000;
    gpio_set_level(GREEN_LED_GPIO, on ? 0 : 1);
    if (!receiving) gpio_set_level(RED_LED_GPIO, 1);
}

static void sensor_int_init(void)
{
    const gpio_config_t int_config = {
        .pin_bit_mask = 1ULL << SENSOR_INT_GPIO,
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    /* INT is an open-drain 1.8 V signal; the level translator/module provides
     * the pull-up, so do not enable an ESP32 3.3 V pull-up here. */
    ESP_ERROR_CHECK(gpio_config(&int_config));
}

static void beep_off(void)
{
    /* The board's BEEP input is active high. Drive it low before any sensor
     * initialization so the alarm stays silent during bring-up. */
    const gpio_config_t beep_config = {
        .pin_bit_mask = 1ULL << BEEP_GPIO,
        .mode = GPIO_MODE_OUTPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&beep_config));
    ESP_ERROR_CHECK(gpio_set_level(BEEP_GPIO, 0));
}

static esp_err_t sensor_bus_init(i2c_master_bus_handle_t *bus_handle)
{
    const i2c_master_bus_config_t bus_config = {
        .i2c_port = SENSOR_I2C_PORT,
        .sda_io_num = CONFIG_FALLINGDETECT_I2C_SDA_GPIO,
        .scl_io_num = CONFIG_FALLINGDETECT_I2C_SCL_GPIO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        /* The pull-ups belong on the 1.8 V side of the level translator. */
        .flags.enable_internal_pullup = false,
    };

    return i2c_new_master_bus(&bus_config, bus_handle);
}

static esp_err_t sensor_device_init(i2c_master_bus_handle_t bus_handle,
                                    VL53L8CX_Configuration *device)
{
    const i2c_device_config_t device_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = SENSOR_I2C_ADDRESS,
        .scl_speed_hz = CONFIG_FALLINGDETECT_I2C_FREQUENCY_HZ,
    };

    device->platform.bus_config.i2c_port = SENSOR_I2C_PORT;
    device->platform.bus_config.sda_io_num = CONFIG_FALLINGDETECT_I2C_SDA_GPIO;
    device->platform.bus_config.scl_io_num = CONFIG_FALLINGDETECT_I2C_SCL_GPIO;
    device->platform.bus_config.clk_source = I2C_CLK_SRC_DEFAULT;
    device->platform.bus_config.glitch_ignore_cnt = 7;
    device->platform.bus_config.flags.enable_internal_pullup = false;

    if (CONFIG_FALLINGDETECT_SENSOR_RESET_GPIO >= 0) {
        device->platform.reset_gpio = (gpio_num_t)CONFIG_FALLINGDETECT_SENSOR_RESET_GPIO;
    } else {
        device->platform.reset_gpio = GPIO_NUM_NC;
    }

    return i2c_master_bus_add_device(bus_handle, &device_config,
                                     &device->platform.handle);
}

static bool is_valid_measurement(uint8_t status)
{
    /* ST defines status 5 and 9 as valid ranging results. */
    return status == 5U || status == 9U;
}

#ifdef CONFIG_FALLINGDETECT_RAW_FRAME_LOG
static void print_frame(const VL53L8CX_Configuration *device,
                        const VL53L8CX_ResultsData *results)
{
    uint16_t valid_count = 0;

    printf("tof frame=%" PRIu32 " distances_mm=[", (uint32_t)device->streamcount);
    for (uint8_t zone = 0; zone < SENSOR_ZONE_COUNT; ++zone) {
        const uint8_t index = (uint8_t)(VL53L8CX_NB_TARGET_PER_ZONE * zone);
        const int16_t distance = results->distance_mm[index];
        const uint8_t status = results->target_status[index];

        if (is_valid_measurement(status)) {
            ++valid_count;
        }
        if (zone != 0) {
            putchar(',');
        }
        if (is_valid_measurement(status) && distance >= 0) {
            printf("%d", distance);
        } else {
            printf("-1");
        }
    }
    printf("] valid=%u temp=%dC\n", (unsigned)valid_count, results->silicon_temp_degc);
}

#endif

#ifdef CONFIG_FALLINGDETECT_RPS_ENABLED
static bool rps_ready;
static rps_label_t rps_candidate = RPS_UNKNOWN;
static uint8_t rps_consecutive;

static void recognize_gesture(const VL53L8CX_Configuration *device,
                              const VL53L8CX_ResultsData *results)
{
    if (!rps_ready) return;
    int16_t distances[64];
    for (int zone = 0; zone < 64; ++zone) {
        int index = VL53L8CX_NB_TARGET_PER_ZONE * zone;
        distances[zone] = is_valid_measurement(results->target_status[index]) && results->distance_mm[index] >= 0
                             ? results->distance_mm[index] : -1;
    }
    int64_t start = esp_timer_get_time();
    rps_result_t prediction;
    rps_predict(distances, &prediction);
    int64_t elapsed = esp_timer_get_time() - start;
    if (prediction.label == RPS_UNKNOWN) {
        rps_candidate = RPS_UNKNOWN; rps_consecutive = 0;
    } else if (prediction.label != rps_candidate) {
        rps_candidate = prediction.label; rps_consecutive = 1;
    } else if (rps_consecutive < 3) {
        ++rps_consecutive;
    }
    rps_label_t stable = rps_consecutive >= 3 ? rps_candidate : RPS_UNKNOWN;
    ble_send_gesture((uint16_t)device->streamcount, &prediction, stable);
    printf("gesture frame=%u label=%s raw=%s confidence=%.3f probs=[%.3f,%.3f,%.3f] valid=%u near=%u inference_us=%" PRId64 "\n",
           (unsigned)device->streamcount, rps_label_name(stable), rps_label_name(prediction.raw_label),
           (double)prediction.confidence, (double)prediction.probabilities[0],
           (double)prediction.probabilities[1], (double)prediction.probabilities[2],
           (unsigned)prediction.valid_zones, (unsigned)prediction.near_zones, elapsed);
}
#endif

void app_main(void)
{
    i2c_master_bus_handle_t bus_handle = NULL;
    /* Driver buffers persist for the lifetime of app_main. Keep them off the
     * main-task stack: diagnostic snapshots must not crowd LSTM self-test and
     * logging call frames during startup. */
    static VL53L8CX_Configuration sensor;
    static VL53L8CX_ResultsData results;
    uint8_t status = 0;
    uint8_t is_alive = 0;
    uint8_t is_ready = 0;

    #ifdef CONFIG_FALLINGDETECT_RPS_ENABLED
    float rps_error;
    rps_ready = rps_self_test(&rps_error);
    ESP_LOGI(TAG, "RPS CNN self-test %s max_probability_error=%.8f", rps_ready ? "PASS" : "FAIL", (double)rps_error);
    #endif

    beep_off();
    indicators_off();
    #ifdef CONFIG_FALLINGDETECT_FALL_ENABLED
    float fall_error;
    fall_reset(&fall_state);
    fall_ready = fall_self_test(&fall_error);
    ESP_LOGI(TAG, "Physics-aware LSTM self-test %s max_probability_error=%.8f window=%d hz=10 down=heatmap+Y",
             fall_ready ? "PASS" : "FAIL", (double)fall_error, FALL_WINDOW_FRAMES);
#if FALL_USE_INT8
    ESP_LOGI(TAG, "Runtime: feature_v2 mixed_INT8/IRAM DRAM_weights; acquire=CPU0 infer=CPU1; UDP nonblocking");
#else
    ESP_LOGI(TAG, "Runtime: feature_v2 matvec4/IRAM DRAM_weights; acquire=CPU0 infer=CPU1; UDP nonblocking");
#endif
    fall_frame_queue = xQueueCreate(FALL_FRAME_QUEUE_LENGTH, sizeof(fall_frame_t));
    ESP_ERROR_CHECK(fall_frame_queue == NULL ? ESP_ERR_NO_MEM : ESP_OK);
    BaseType_t task_rc = xTaskCreatePinnedToCore(fall_inference_task, "fall_inference", 8192,
                                                  NULL, 6, NULL, 1);
    ESP_ERROR_CHECK(task_rc == pdPASS ? ESP_OK : ESP_ERR_NO_MEM);
    #endif
    sensor_int_init();
    ESP_ERROR_CHECK(ble_init());
    xTaskCreate(wifi_config_task, "wifi_config", 4096, NULL, 5, NULL);
    ESP_ERROR_CHECK(sensor_bus_init(&bus_handle));
    ESP_ERROR_CHECK(sensor_device_init(bus_handle, &sensor));

    if (CONFIG_FALLINGDETECT_SENSOR_RESET_GPIO >= 0) {
        ESP_LOGI(TAG, "resetting VL53L8CX on GPIO %d",
                 CONFIG_FALLINGDETECT_SENSOR_RESET_GPIO);
        status = VL53L8CX_Reset_Sensor(&sensor.platform);
        if (status != 0) {
            ESP_LOGE(TAG, "sensor reset failed: %u", (unsigned)status);
            return;
        }
    }

    status = vl53l8cx_is_alive(&sensor, &is_alive);
    if (status != 0 || is_alive == 0) {
        ESP_LOGE(TAG, "VL53L8CX not detected (status=%u, alive=%u)",
                 (unsigned)status, (unsigned)is_alive);
        return;
    }

    status = vl53l8cx_init(&sensor);
    if (status != 0) {
        ESP_LOGE(TAG, "VL53L8CX ULD init failed: %u", (unsigned)status);
        return;
    }

    status = vl53l8cx_set_resolution(&sensor, SENSOR_ZONE_COUNT);
    if (status != 0) {
        ESP_LOGE(TAG, "set 8x8 resolution failed: %u", (unsigned)status);
        return;
    }

    status = vl53l8cx_set_ranging_frequency_hz(
        &sensor, CONFIG_FALLINGDETECT_RANGING_FREQUENCY_HZ);
    if (status != 0) {
        ESP_LOGE(TAG, "set ranging frequency failed: %u", (unsigned)status);
        return;
    }

    status = vl53l8cx_set_ranging_mode(&sensor, VL53L8CX_RANGING_MODE_CONTINUOUS);
    if (status != 0) {
        ESP_LOGE(TAG, "set continuous mode failed: %u", (unsigned)status);
        return;
    }

#ifdef FALL_SENSOR_DIAGNOSTICS
    uint8_t read_status[6],read_resolution=0,read_hz=0,read_mode=0,read_sharpener=0,read_order=0;
    uint32_t read_integration=0;
    read_status[0]=vl53l8cx_get_resolution(&sensor,&read_resolution);
    read_status[1]=vl53l8cx_get_ranging_frequency_hz(&sensor,&read_hz);
    read_status[2]=vl53l8cx_get_ranging_mode(&sensor,&read_mode);
    read_status[3]=vl53l8cx_get_integration_time_ms(&sensor,&read_integration);
    read_status[4]=vl53l8cx_get_sharpener_percent(&sensor,&read_sharpener);
    read_status[5]=vl53l8cx_get_target_order(&sensor,&read_order);
    printf("sensor_config {\"resolution\":%u,\"hz\":%u,\"mode\":%u,\"integration_register_ms\":%" PRIu32
           ",\"sharpener_percent\":%u,\"target_order\":%u,\"read_status\":[%u,%u,%u,%u,%u,%u]}\n",
           (unsigned)read_resolution,(unsigned)read_hz,(unsigned)read_mode,read_integration,(unsigned)read_sharpener,(unsigned)read_order,
           (unsigned)read_status[0],(unsigned)read_status[1],(unsigned)read_status[2],(unsigned)read_status[3],(unsigned)read_status[4],(unsigned)read_status[5]);
    ESP_LOGI(TAG,"ToF diagnostics: at most 60 snapshots, 1 Hz on CPU1; ranging settings and validity policy unchanged");
#endif
    status = vl53l8cx_start_ranging(&sensor);
    if (status != 0) {
        ESP_LOGE(TAG, "start ranging failed: %u", (unsigned)status);
        return;
    }

    ESP_LOGI(TAG, "VL53L8CX ready: 8x8 at %d Hz, I2C %d Hz",
             CONFIG_FALLINGDETECT_RANGING_FREQUENCY_HZ,
             CONFIG_FALLINGDETECT_I2C_FREQUENCY_HZ);
    ESP_LOGI(TAG, "Runtime memory: free_heap=%" PRIu32 " min_free_heap=%" PRIu32,
             esp_get_free_heap_size(),esp_get_minimum_free_heap_size());

    ESP_LOGI(TAG, "Running LED: GPIO5 green, 500 ms on / 500 ms off while receiving frames");
    int64_t last_frame_us = 0;
    while (true) {
        status = vl53l8cx_check_data_ready(&sensor, &is_ready);
        if (status != 0) {
            ESP_LOGW(TAG, "check data ready failed: %u", (unsigned)status);
        } else if (is_ready != 0) {
            status = vl53l8cx_get_ranging_data(&sensor, &results);
            if (status != 0) {
                ESP_LOGW(TAG, "read ranging data failed: %u", (unsigned)status);
            } else {
                last_frame_us = esp_timer_get_time();
                #ifdef CONFIG_FALLINGDETECT_FALL_ENABLED
                enqueue_fall_frame(&sensor, &results, (uint32_t)(last_frame_us / 1000));
                #endif
                #ifdef CONFIG_FALLINGDETECT_RPS_ENABLED
                recognize_gesture(&sensor, &results);
                #endif
                #ifdef CONFIG_FALLINGDETECT_RAW_FRAME_LOG
                print_frame(&sensor, &results);
                #endif
                wifi_send_frame(&sensor, &results);
            }
        }
        update_running_led(last_frame_us);
        vTaskDelay(pdMS_TO_TICKS(5));
    }
}
