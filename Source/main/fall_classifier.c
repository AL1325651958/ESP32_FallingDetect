#include "fall_classifier.h"
#if FALL_USE_INT8
#include "fall_model_data_int8.h"
#else
#include "fall_model_data.h"
#endif
#include <float.h>
#include <math.h>
#include <string.h>
#ifdef ESP_PLATFORM
#include "esp_attr.h"
#define FALL_HOT IRAM_ATTR
#else
#define FALL_HOT
#endif

_Static_assert(FALL_WINDOW_FRAMES == FALL_WINDOW, "window mismatch");
_Static_assert(FALL_FEATURE_VERSION == 2, "retrain weights for masked/common-support features");
static float FALL_HOT sigmoid(float x) { return x >= 0 ? 1.0f/(1.0f+expf(-x)) : expf(x)/(1.0f+expf(x)); }

#ifdef FALL_TESTING
/* Host-only injection exercises decisions even when dropout fools model scores. */
static bool test_override;
static float test_probability[3];
void fall_test_scores(float motion, float fall, float still)
{
    test_override = motion >= 0;
    test_probability[0]=motion;test_probability[1]=fall;test_probability[2]=still;
}
#endif

#if FALL_USE_INT8
/* Per-output symmetric INT8 weights and per-call symmetric INT8 activations.
 * At most 192 products of [-127,127] fit comfortably in int32_t. LSTM state,
 * biases, physical features, sigmoid, tanh and softmax remain float32. */
static void FALL_HOT matvec_int8(const int8_t *weights,const float *weight_scale,
                                const float *input,unsigned rows,unsigned columns,float *output)
{
    int8_t quantized[192];float peak=0;
    for(unsigned k=0;k<columns;++k)peak=fmaxf(peak,fabsf(input[k]));
    if(peak==0)return;
    float inverse=127.0f/peak,scale=peak/127.0f;
    for(unsigned k=0;k<columns;++k){
        float v=input[k]*inverse;
        int q=v>=0?(int)(v+.5f):(int)(v-.5f);
        quantized[k]=(int8_t)(q>127?127:q< -127?-127:q);
    }
    unsigned row=0;
    for(;row+3<rows;row+=4){
        const int8_t *w0=weights+row*columns,*w1=w0+columns,*w2=w1+columns,*w3=w2+columns;
        int32_t a=0,b=0,c=0,d=0;
        for(unsigned k=0;k<columns;++k){int32_t x=quantized[k];a+=w0[k]*x;b+=w1[k]*x;c+=w2[k]*x;d+=w3[k]*x;}
        output[row]+=(float)a*(weight_scale[row]*scale);
        output[row+1]+=(float)b*(weight_scale[row+1]*scale);
        output[row+2]+=(float)c*(weight_scale[row+2]*scale);
        output[row+3]+=(float)d*(weight_scale[row+3]*scale);
    }
    for(;row<rows;++row){int32_t a=0;for(unsigned k=0;k<columns;++k)a+=weights[row*columns+k]*(int32_t)quantized[k];output[row]+=(float)a*(weight_scale[row]*scale);}
}
#define FALL_MATVEC(w,x,r,c,y) matvec_int8(w,w##_scale,x,r,c,y)
#else
/* Reuse each input load for four independent output accumulators. Preserve the
 * original per-neuron summation order, rather than using approximate math. */
static void FALL_HOT matvec_accumulate(const float *weights,const float *input,
                                      unsigned rows,unsigned columns,float *output)
{
    unsigned row=0;
    for(;row+3<rows;row+=4){
        const float *w0=weights+row*columns,*w1=w0+columns,*w2=w1+columns,*w3=w2+columns;
        float a=output[row],b=output[row+1],c=output[row+2],d=output[row+3];
        for(unsigned k=0;k<columns;++k){float x=input[k];a+=w0[k]*x;b+=w1[k]*x;c+=w2[k]*x;d+=w3[k]*x;}
        output[row]=a;output[row+1]=b;output[row+2]=c;output[row+3]=d;
    }
    for(;row<rows;++row){float a=output[row];for(unsigned k=0;k<columns;++k)a+=weights[row*columns+k]*input[k];output[row]=a;}
}
#define FALL_MATVEC(w,x,r,c,y) matvec_accumulate(w,x,r,c,y)
#endif

/* Feature order and stencils match NN/train_physics_fall.py. */
static void FALL_HOT spatial_features(const int16_t *raw,float x[128],float local[192],float stats[12],
                             const float previous[128],const float previous_stats[12],bool first)
{
    float grid[64],mass=0,mx=0,my=0,md=0,valid=0;
    for(unsigned z=0;z<64;++z){
        x[z+64]=raw[z]>=0?1.0f:0.0f;
        x[z]=raw[z]<0?0.0f:fminf((float)raw[z],FALL_DISTANCE_SCALE)/FALL_DISTANCE_SCALE;
        grid[z]=x[z]; float q=1-grid[z],w=x[z+64]*q*q;
        mass+=w;mx+=w*((float)(z%8)*2.0f/7.0f-1.0f);my+=w*((float)(z/8)*2.0f/7.0f-1.0f);md+=grid[z];valid+=x[z+64];
    }
    mass=fmaxf(mass,1e-6f);float cx=mx/mass,cy=my/mass,sx=0,sy=0;
    for(unsigned z=0;z<64;++z){float q=1-grid[z],w=x[z+64]*q*q;
        float dx=(float)(z%8)*2.0f/7.0f-1.0f-cx,dy=(float)(z/8)*2.0f/7.0f-1.0f-cy;
        sx+=w*dx*dx;sy+=w*dy*dy;}
    stats[0]=cx;stats[1]=cy;stats[2]=sqrtf(fmaxf(sx/mass,0));stats[3]=sqrtf(fmaxf(sy/mass,0));
    stats[4]=md/fmaxf(valid,1);stats[5]=valid/64;
    float energy=0,common=0,old_valid=0,old_mass=0,new_mass=0;
    float old_x=0,old_y=0,new_x=0,new_y=0,old_mean=0,new_mean=0;
    for(unsigned z=0;z<64;++z){
        old_valid+=previous[64+z];
        if(x[z+64]<=0||previous[64+z]<=0)continue;
        float px=(float)(z%8)*2.0f/7.0f-1.0f,py=(float)(z/8)*2.0f/7.0f-1.0f;
        float oq=1-previous[z],nq=1-grid[z],ow=oq*oq,nw=nq*nq;
        old_mass+=ow;new_mass+=nw;old_x+=ow*px;old_y+=ow*py;new_x+=nw*px;new_y+=nw*py;
        old_mean+=previous[z];new_mean+=grid[z];energy+=fabsf(grid[z]-previous[z]);common++;
    }
    bool reliable=!first&&common>=FALL_MIN_VALID&&2*common>=fmaxf(valid,old_valid);
    stats[6]=reliable?new_x/fmaxf(new_mass,1e-6f)-old_x/fmaxf(old_mass,1e-6f):0;
    stats[7]=reliable?new_y/fmaxf(new_mass,1e-6f)-old_y/fmaxf(old_mass,1e-6f):0;
    stats[8]=reliable?(new_mean-old_mean)/fmaxf(common,1):0;
    stats[9]=reliable?energy/fmaxf(common,1):0;
    stats[10]=first?0:previous_stats[10]+stats[6];stats[11]=first?0:previous_stats[11]+stats[7];
    for(unsigned y=0;y<8;++y)for(unsigned xx=0;xx<8;++xx){unsigned z=y*8+xx;
        float gx=xx==0?grid[z+1]-grid[z]:xx==7?grid[z]-grid[z-1]:(grid[z+1]-grid[z-1])*.5f;
        float gy=y==0?grid[z+8]-grid[z]:y==7?grid[z]-grid[z-8]:(grid[z+8]-grid[z-8])*.5f;
        unsigned xl=xx==0?z:z-1,xr=xx==7?z:z+1;
        unsigned yu=y==0?z:z-8,yd=y==7?z:z+8;
        local[z]=x[64+z]>0&&x[64+xl]>0&&x[64+xr]>0?gx:0;
        local[64+z]=x[64+z]>0&&x[64+yu]>0&&x[64+yd]>0?gy:0;
        local[128+z]=!first&&x[64+z]>0&&previous[64+z]>0?grid[z]-previous[z]:0;
    }
}

void fall_features_for_test(const int16_t *frames,unsigned count,float *out)
{
    float x[128],local[192],stats[12],prev[128]={0},prevstats[12]={0};
    for(unsigned t=0;t<count;++t){
        spatial_features(&frames[t*64],x,local,stats,prev,prevstats,t==0);
        memcpy(out+t*332,x,128*sizeof(float));memcpy(out+t*332+128,local,192*sizeof(float));
        memcpy(out+t*332+320,stats,12*sizeof(float));memcpy(prev,x,sizeof(x));memcpy(prevstats,stats,sizeof(stats));
    }
}

static void FALL_HOT predict_ring(const int16_t frames[FALL_WINDOW][64],unsigned first,unsigned count,float probabilities[3])
{
#ifdef FALL_TESTING
    if(test_override){memcpy(probabilities,test_probability,sizeof(test_probability));return;}
#endif
    float h[FALL_HIDDEN]={0},c[FALL_HIDDEN]={0},maximum[FALL_HIDDEN];
    float x[128],local[192],stats[12],prev[128]={0},prevstats[12]={0};
    float projected[FALL_PROJECT+FALL_LOCAL+12],gates[4*FALL_HIDDEN];
    for(unsigned j=0;j<FALL_HIDDEN;++j)maximum[j]=-FLT_MAX;
    for(unsigned t=0;t<count;++t){
        spatial_features(frames[(first+t)%FALL_WINDOW],x,local,stats,prev,prevstats,t==0);
        memcpy(projected,fall_spatial_bias,FALL_PROJECT*sizeof(float));
        FALL_MATVEC(fall_spatial_weight,x,FALL_PROJECT,128,projected);
        memcpy(projected+FALL_PROJECT,fall_local_bias,FALL_LOCAL*sizeof(float));
        FALL_MATVEC(fall_local_weight,local,FALL_LOCAL,192,projected+FALL_PROJECT);
        for(unsigned j=0;j<FALL_PROJECT+FALL_LOCAL;++j)projected[j]=fmaxf(projected[j],0);
        memcpy(projected+FALL_PROJECT+FALL_LOCAL,stats,sizeof(stats));
        for(unsigned j=0;j<4*FALL_HIDDEN;++j)gates[j]=fall_lstm_bias_ih_l0[j]+fall_lstm_bias_hh_l0[j];
        FALL_MATVEC(fall_lstm_weight_ih_l0,projected,4*FALL_HIDDEN,FALL_PROJECT+FALL_LOCAL+12,gates);
        FALL_MATVEC(fall_lstm_weight_hh_l0,h,4*FALL_HIDDEN,FALL_HIDDEN,gates);
        for(unsigned j=0;j<FALL_HIDDEN;++j){c[j]=sigmoid(gates[FALL_HIDDEN+j])*c[j]+sigmoid(gates[j])*tanhf(gates[2*FALL_HIDDEN+j]);h[j]=sigmoid(gates[3*FALL_HIDDEN+j])*tanhf(c[j]);maximum[j]=fmaxf(maximum[j],h[j]);}
        memcpy(prev,x,sizeof(x));memcpy(prevstats,stats,sizeof(stats));
    }
    float dense[FALL_DENSE];
#if FALL_USE_INT8
    float pooled[2*FALL_HIDDEN];
    memcpy(pooled,h,sizeof(h));memcpy(pooled+FALL_HIDDEN,maximum,sizeof(maximum));
    memcpy(dense,fall_head_bias,sizeof(dense));
    FALL_MATVEC(fall_head_weight,pooled,FALL_DENSE,2*FALL_HIDDEN,dense);
    for(unsigned j=0;j<FALL_DENSE;++j)dense[j]=fmaxf(dense[j],0);
    float logits[3],maxlog=-FLT_MAX;
    memcpy(logits,fall_out_bias,sizeof(logits));
    FALL_MATVEC(fall_out_weight,dense,3,FALL_DENSE,logits);
    for(unsigned j=0;j<3;++j)maxlog=fmaxf(maxlog,logits[j]);
#else
    for(unsigned j=0;j<FALL_DENSE;++j){float a=fall_head_bias[j];for(unsigned k=0;k<FALL_HIDDEN;++k){a+=fall_head_weight[j*2*FALL_HIDDEN+k]*h[k];a+=fall_head_weight[j*2*FALL_HIDDEN+FALL_HIDDEN+k]*maximum[k];}dense[j]=fmaxf(a,0);}
    float logits[3],maxlog=-FLT_MAX;
    for(unsigned j=0;j<3;++j){float a=fall_out_bias[j];for(unsigned k=0;k<FALL_DENSE;++k)a+=fall_out_weight[j*FALL_DENSE+k]*dense[k];logits[j]=a;maxlog=fmaxf(maxlog,a);}
#endif
    float sum=0;for(unsigned j=0;j<3;++j){probabilities[j]=expf(logits[j]-maxlog);sum+=probabilities[j];}
    for(unsigned j=0;j<3;++j)probabilities[j]/=sum;
}

float fall_predict_sequence(const int16_t *distances,unsigned frames,float probabilities[3])
{
    if(frames==0||frames>FALL_WINDOW)return NAN;
    predict_ring((const int16_t (*)[64])distances,0,frames,probabilities);return probabilities[1];
}
float fall_predict_window(const int16_t distances[FALL_WINDOW][64],float probabilities[3])
{return fall_predict_sequence(&distances[0][0],FALL_WINDOW,probabilities);}

/* These are conservative runtime guards, not trained occupancy classes. */
#define FALL_EVIDENCE_AGE_MS 2500U
#define FALL_HISTORY_MASK ((1U << FALL_WINDOW) - 1U)

static unsigned evidence_count(uint32_t bits)
{
    unsigned n=0;for(;bits;bits&=bits-1)++n;return n;
}

static void clear_window(fall_state_t *s)
{
    s->count=s->next=s->high_count=s->low_count=s->predict_phase=0;
    s->motion_history=s->downward_history=0;
    s->raw=FALL_UNKNOWN;
    memset(s->probabilities,0,sizeof(s->probabilities));
}

/* Missing pixels never count as movement. Compare the same measured zones;
 * require several changing points and sufficient overlap, not component splits
 * or a centroid computed from different masks. Distances are clipped exactly
 * as in the unchanged model feature extractor. */
static void observe_motion(fall_state_t *s,const int16_t raw[64],uint32_t now_ms)
{
    bool moved=false,downward=false;
    if(s->count){
        const int16_t *previous=s->frames[(s->next+FALL_WINDOW-1)%FALL_WINDOW];
        unsigned common=0,old_valid=0,new_valid=0,mask_changes=0;
        int8_t direction[64]={0};
        float old_mass=0,new_mass=0,old_y=0,new_y=0;
        for(unsigned z=0;z<64;++z){
            old_valid+=previous[z]>=0;new_valid+=raw[z]>=0;
            mask_changes+=(previous[z]>=0)!=(raw[z]>=0);
            if(previous[z]<0||raw[z]<0)continue;
            float old=fminf(previous[z],FALL_DISTANCE_SCALE),current=fminf(raw[z],FALL_DISTANCE_SCALE);
            float delta=current-old;common++;
            if(fabsf(delta)>=40)direction[z]=delta>0?1:-1;
            float oq=1-old/FALL_DISTANCE_SCALE,nq=1-current/FALL_DISTANCE_SCALE;
            float ow=oq*oq,nw=nq*nq,y=(float)(z/8)*2/7-1;
            old_mass+=ow;new_mass+=nw;old_y+=ow*y;new_y+=nw*y;
        }
        bool overlap=common>=FALL_MIN_VALID&&2*common>=(old_valid>new_valid?old_valid:new_valid);
        /* Local coherent changes can describe a small person region without
         * being diluted by a large, valid, stationary background. */
        if(overlap)for(unsigned z=0;z<64;++z){
            int8_t d=direction[z];if(!d)continue;
            unsigned neighbors=0;
            if(z%8)neighbors+=direction[z-1]==d;
            if(z%8<7)neighbors+=direction[z+1]==d;
            if(z>=8)neighbors+=direction[z-8]==d;
            if(z<56)neighbors+=direction[z+8]==d;
            if(neighbors>=2){moved=true;break;}
        }
        downward=moved&&new_y/fmaxf(new_mass,1e-6f)-old_y/fmaxf(old_mass,1e-6f)>=.015f;
        /* Earlier ordinary movement must not license a new alert caused only
         * by a later hole. Keep established alerts/candidates separately. */
        if(mask_changes&&!moved)s->motion_history=s->downward_history=0;
    }
    s->motion_history=((s->motion_history<<1)|moved)&FALL_HISTORY_MASK;
    s->downward_history=((s->downward_history<<1)|downward)&FALL_HISTORY_MASK;
    if(moved){s->has_motion=true;s->last_motion_ms=now_ms;}
    if(downward&&evidence_count(s->downward_history)>=2){s->has_downward=true;s->last_downward_ms=now_ms;}
}

void fall_reset(fall_state_t *s)
{
    memset(s,0,sizeof(*s));s->stable=s->raw=s->held_alert=FALL_UNKNOWN;
}
void fall_push(fall_state_t *s,const int16_t distances[64],uint32_t now_ms,fall_result_t *r)
{
    memset(r,0,sizeof(*r));r->raw=r->stable=-1;
    for(unsigned z=0;z<64;++z)r->valid_zones+=distances[z]>=0;
    uint32_t dt=now_ms-s->last_ms;
    if(s->has_time&&(dt<FALL_GAP_MIN_MS||dt>FALL_GAP_MAX_MS)){
        int8_t held=s->held_alert;
        bool downward=s->has_downward,candidate=s->has_candidate;
        uint32_t downward_ms=s->last_downward_ms,candidate_ms=s->last_candidate_ms;
        /* A sampling gap cannot clear an existing alert or create motion. */
        fall_reset(s);s->stable=s->held_alert=held;
        s->has_downward=downward;s->last_downward_ms=downward_ms;
        s->has_candidate=candidate;s->last_candidate_ms=candidate_ms;
    }
    s->last_ms=now_ms;s->has_time=true;
    if(r->valid_zones<FALL_MIN_VALID){
        if(!s->observation_lost){
            bool descending=s->has_downward&&(uint32_t)(now_ms-s->last_downward_ms)<=FALL_EVIDENCE_AGE_MS;
            bool candidate=s->has_candidate&&(uint32_t)(now_ms-s->last_candidate_ms)<=FALL_EVIDENCE_AGE_MS;
            if(s->held_alert!=FALL_DETECTED&&(descending||candidate))s->held_alert=FALL_SUSPECTED;
            clear_window(s);s->observation_lost=true;
        }
        r->raw=FALL_OBSERVATION_LOST;
        r->stable=s->held_alert>=0?s->held_alert:FALL_OBSERVATION_LOST;
        s->stable=r->stable;
        return;
    }
    if(s->observation_lost)s->stable=s->held_alert;
    s->observation_lost=false;
    observe_motion(s,distances,now_ms);
    memcpy(s->frames[s->next],distances,sizeof(s->frames[0]));
    s->next=(s->next+1)%FALL_WINDOW;if(s->count<FALL_WINDOW)++s->count;r->buffered_frames=s->count;
    if(s->count<FALL_WINDOW){r->stable=s->held_alert;return;}
    if(s->predict_phase){
        s->predict_phase=0;r->raw=s->raw;r->stable=s->stable;
        memcpy(r->probability,s->probabilities,sizeof(r->probability));return;
    }
    s->predict_phase=1;r->prediction_updated=true;
    predict_ring(s->frames,s->next,FALL_WINDOW,r->probability);
    unsigned best=0;for(unsigned k=1;k<3;++k)if(r->probability[k]>r->probability[best])best=k;r->raw=(int8_t)best;
    bool physical=evidence_count(s->motion_history)>=2&&s->has_motion&&
                  (uint32_t)(now_ms-s->last_motion_ms)<=FALL_EVIDENCE_AGE_MS;
    if(physical&&r->probability[1]>=.5f){s->has_candidate=true;s->last_candidate_ms=now_ms;}
    if(r->probability[1]>=.8f&&physical){s->low_count=0;if(s->high_count<2)++s->high_count;if(s->high_count>=2)s->stable=s->held_alert=FALL_DETECTED;}
    else if(r->probability[1]<=.3f){s->high_count=0;if(s->low_count<3)++s->low_count;if(s->low_count>=3){s->stable=r->raw;s->held_alert=FALL_UNKNOWN;s->has_candidate=s->has_downward=false;}}
    else s->high_count=s->low_count=0;
    r->stable=s->stable;s->raw=r->raw;
    memcpy(s->probabilities,r->probability,sizeof(s->probabilities));
}
bool fall_self_test(float *max_error)
{
    *max_error=0;for(unsigned i=0;i<FALL_SELF_TEST_COUNT;++i){float p[3];
        fall_predict_sequence(&fall_test_raw[i*FALL_WINDOW*64],fall_test_lengths[i],p);
        float e=0;for(unsigned k=0;k<3;++k){if(!isfinite(p[k]))return false;e=fmaxf(e,fabsf(p[k]-fall_test_probability[i*3+k]));}
        if(!isfinite(e)) return false;
        *max_error=fmaxf(*max_error,e);
    }
    return *max_error<.0001f;
}
