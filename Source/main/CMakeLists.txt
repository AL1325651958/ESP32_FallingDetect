idf_component_register(
    SRCS "main.c" "rps_classifier.c" "fall_classifier.c"
    INCLUDE_DIRS "."
    REQUIRES vl53l8cx esp_driver_gpio esp_driver_i2c esp_wifi esp_netif bt nvs_flash esp_app_format esp_timer
)

option(FALL_USE_INT8 "Use mixed INT8 fall model (float remains default)" OFF)
if(FALL_USE_INT8)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FALL_USE_INT8=1)
endif()

option(FALL_SENSOR_DIAGNOSTICS "Log raw ToF quality once per second without changing ranging settings" OFF)
if(FALL_SENSOR_DIAGNOSTICS)
    target_compile_definitions(${COMPONENT_LIB} PRIVATE FALL_SENSOR_DIAGNOSTICS=1)
endif()
