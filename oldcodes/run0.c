/*
============================================================
说明
------------------------------------------------------------
1）实现 forward() 中，逐层循环 for (int l = 0; l < p->n_layers; l++) 代码，
   保留了中文提示，请按提示补全代码逻辑；
2）其余代码尽量保持不变，确保可以编译运行。

建议完成顺序：
- 先理解整个推理的流程；
- 补齐 RMSNorm、量化/反量化、矩阵乘法、Softmax函数、SwiGLU等；
- 再根据forward() 提示，调用上述函数；
- 通过打印中间变量或断点调试验证每一步的张量形状与数值范围是否合理。

祝顺利！
============================================================
*/

/* 使用纯 C 实现的 Llama-2 Transformer 推理（int8 量化前向计算） */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif
// ----------------------------------------------------------------------------
// 全局变量
int GS = 0; // 权重量化使用的全局分组大小

// ----------------------------------------------------------------------------
// Transformer 模型

typedef struct {
    int dim; // Transformer 维度
    int hidden_dim; // FFN 层隐藏维度
    int n_layers; // 层数
    int n_heads; // Query 头数
    int n_kv_heads; // Key/Value 头数（多查询时可小于 n_heads）
    int vocab_size; // 词表大小（字节级，通常为 256）
    int seq_len; // 最大序列长度
} Config;

typedef struct {
    int8_t* q;    // 量化后的值
    float* s; // 量化缩放因子
} QuantizedTensor;

typedef struct {
    // 词嵌入查找表
    QuantizedTensor *q_tokens; // (vocab_size, dim)
    float* token_embedding_table; // 同上，但已反量化为 float

    // RMSNorm 的权重
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // 线性层（矩阵乘法）的权重。注意 dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // 前馈网络（FFN）的权重
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // 最后的 RMSNorm
    float* rms_final_weight; // (dim,)
    // （可选）分类器/输出层权重（用于计算 logits）
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // 当前时间步的激活向量
    float *x; // activation at current time stamp (dim,)
    float *xb; // 相同大小，用于残差分支中的中间结果 (dim,)
    float *xb2; // 额外的缓冲区，使用更方便 (dim,)
    float *hb; // FFN 隐藏层维度的缓冲区 (hidden_dim,)
    float *hb2; // FFN 隐藏层维度的缓冲区 (hidden_dim,)
    QuantizedTensor xq; // 量化后的 x（长度为 dim）
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // 注意力分数/权重缓冲区（形状：n_heads × seq_len）
    float *logits; // 输出 logits
    // 键值（KV）缓存
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // 模型结构的超参数（蓝图）
    TransformerWeights weights; // 模型的权重
    RunState state; // 前向传播过程中使用的各类缓冲区
    // 一些用于清理内存映射的状态
    int fd; // 用于内存映射的文件描述符
    float* data; // 内存映射的数据指针
    ssize_t file_size; // 检查点文件的大小（以字节为单位）
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // 使用 calloc（而非 malloc），便于内存检查工具（如 valgrind）
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = calloc(p->dim, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim, sizeof(float)) };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    // 检查所有内存分配是否成功
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits || !s->key_cache
     || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// 量化与反量化函数

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // 在当前分组中找到最大绝对值
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // 计算并写入缩放因子
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // 计算并写入量化后的数值
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

/* 从 ptr 指向的内存开始，初始化一个包含 n 个量化张量的数组（每个张量有 size_each 个元素） */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* 映射量化后的 int8 数值*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
        /* map scale factors */
        res[i].s = (float*)p;
        p = (float*)p + size_each / GS;
    }
    *ptr = p; // 将指针 ptr 前移到当前位置
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // 先映射保留为 fp32 的参数（各层 RMSNorm 一维权重）
    float* fptr = (float*) ptr; // 将指针转为 float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // 以下映射所有量化权重
    ptr = (void*)fptr; // 将指针转回 void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    // 反量化词嵌入表
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }

    uint32_t magic_number;
    if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }

    int version;
    if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }
    int header_size = 256; 

    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

    uint8_t shared_classifier; // 指示分类器权重是否与嵌入表共享的字节标志
    if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }
    int group_size; // 量化所用的分组大小
    if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
    GS = group_size; // 设为全局变量，供多处使用
    // 获取文件大小
    fseek(file, 0, SEEK_END); // 将文件指针移到文件末尾
    *file_size = ftell(file); // 获取文件大小（单位：字节）
    fclose(file);
    // 将 Transformer 权重以内存映射的方式读入
    *fd = open(checkpoint, O_RDONLY); // 以只读方式打开
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    void* weights_ptr = ((char*)*data) + header_size; // 跳过头部字节（char 为 1 字节）
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// 神经网络模块：Transformer 的核心计算

void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }

}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // 计算：W(d×n) @ x(n) -> xout(d)
    // 矩阵向量相乘
        int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);
            }
            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
            ival = 0;
        }

        xout[i] = val;
    }

}

float* forward(Transformer* transformer, int token, int pos) {

    // 便捷的局部变量
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // 将当前 token 的嵌入拷贝到 x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // 逐层执行前向计算
    for(int l = 0; l < p->n_layers; l++) {

        // 1) 对 x 进行 RMSNorm，结果写入 s->xb
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // 2) 将 s->xb 量化到 s->xq（int8 + 缩放因子）
        quantize(&s->xq, s->xb, dim);


        // 3) 计算当前层的 q, k, v（矩阵乘法）
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // 4) RoPE 旋转位置编码：对每个 head 中的 (i, i+1) 偶/奇通道成对旋转
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; 
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; 
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }
 

        // 5) 将 k / v 写入 KV Cache（第 l 层，第 pos 个时刻）
        int loff = l * p->seq_len * kv_dim; 
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
      

        // 6) 多头注意力：
        //    (a) 对每个头 h，计算与 t=0..pos 的注意力分数（q·k / sqrt(head_size)）
        //    (b) 对分数做 softmax 得到注意力权重
        //    (c) 用权重对对应的 v 做加权求和，结果拼接进 s->xb
        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }
            softmax(att, pos + 1);
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
  
        }

        // 7) 输出投影：量化 s->xb 到 s->xq，然后乘以 w->wo[l] 得到 s->xb2（矩阵乘法）
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // 8) 残差连接：x += s->xb2
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }
        // ==================== 前馈网络（FFN / SwiGLU） ====================
        // 9) 对 x 做 RMSNorm，结果写入 s->xb
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // 10) 量化 s->xb -> s->xq，并计算 w1(x)->hb（矩阵乘法） 与 w3(x)->hb2（矩阵乘法）
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // 11) SwiGLU 非线性：hb = silu(hb) * hb2（其中 silu(x)=x*σ(x)）
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }
        // 12) 量化 hb -> hq，计算 w2(hb) -> s->xb（矩阵乘法）,补充...的维度大小
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l,hidden_dim, dim);

        // 13) 残差连接：x += s->xb
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }


    // 最后的 RMSNorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // 最后的分类器得到 logits
    quantize(&s->xq, x, dim);
    matmul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// BPE 分词器：在字符串与 token 序列之间转换

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; 
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    // 本应将 vocab_size 写入 tokenizer 文件，这里通过参数传入
    t->vocab_size = vocab_size;
    // 为分数与字符串分配空间
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // 延迟初始化
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // 读取分词器文件
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE);}
        if (fread(&len, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0'; // 追加字符串终止符
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // 在 BOS(=1) 之后，SentencePiece 解码会去掉前导空格（参见 PR #89）
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // 注意：有些 token 表示原始字节，形如 '<0x01>'
    // 解析并返回其对应的实际字节
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // 片段可能是原始字节，仅打印可见字符或空白字符
    // 因为其他字节可能是控制码、退格等
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // 不可打印字节，跳过
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // 在词表中高效查找字符串的精确匹配，返回其索引；找不到则返回 -1
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // 将输入字符串编码为 token，写入预分配的 tokens[]
    // bos!=0 表示在前面添加 BOS(=1)；eos!=0 表示在末尾追加 EOS(=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // 延迟分配并排序词表
        t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // 创建临时缓冲区，用于存放两两相邻 token 的合并候选
    // *2 预留拼接空间，+1 结尾空字符，+2 兼容 UTF‑8（当 max_token_length=1 时）
    char* str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // 初始化 token 计数为 0
    *n_tokens = 0;

    // 需要时添加可选的 BOS(=1)
    if (bos) tokens[(*n_tokens)++] = 1;

    // 默认会添加一个“虚拟前缀”
    // 若文本非空，则在最前面添加一个空格 token 作为前缀
    // TODO：一般情况可能并不完全正确，这里保持与参考实现一致
    // 不再深挖 SentencePiece 代码细节
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // 下面处理 UTF‑8 字节序列（规则参考维基百科）
    // 码点与 UTF‑8 转换关系（简述）
    // 起始码点	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // 处理输入字符串的原始 UTF‑8 字节序列
    for (char *c = text; *c != '\0'; c++) {

        // 若当前字节是 ASCII 或起始字节，则重置缓冲区
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // 在 UTF‑8 中，续字节高两位为 10
        // 也即：若该字节不是续字节
        if ((*c & 0xC0) != 0x80) {
            // 该字节要么是起始字节（11...），要么是 ASCII（0x...）
            // => 开始新的码点，重置位置
            str_len = 0;
        }

        // 将当前字节追加到缓冲区
        str_buffer[str_len++] = *c; // ++ 为后缀自增，在本行执行后发生
        str_buffer[str_len] = '\0';

        // 若下一个字节仍为续字节，则继续追加
        // 但为防止溢出，超过上限则停止
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // 当 c+1 不是续字节时，说明已读完一个完整码点
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // 该码点在词表中，作为一个 token 加入
            tokens[(*n_tokens)++] = id;
        } else {
            // 退化为逐字节编码：将每个字节当作一个 token
            // +3 是因为前 3 个词是 <unk>, <s>, </s>
            // 因此字节 token 从索引 3 开始
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // 防止连续的游离 UTF‑8 续字节导致问题
    }

    // 每轮按 vocab_scores 合并得分最高的一对相邻 token
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // 检查 (tokens[i], tokens[i+1]) 是否可合并
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // 该合并对存在于词表，记录其分数与位置
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // 没有可合并的对，合并结束
        }

        // 将相邻对 (best_idx, best_idx+1) 合并为新 token best_id
        tokens[best_idx] = best_id;
        // 删除 best_idx+1 位置的 token，并整体左移一位
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token 序列长度减少
    }

    // 需要时添加可选的 EOS(=2)
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// 采样器：接收 logits 并返回采样到的下一个 token
// 采样方式：贪心（argmax）、随机采样、top-p（核）采样

typedef struct {
    float prob;
    int index;
} ProbIndex; // 用于 top-p 采样时排序的结构体

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // 仅在 top‑p 采样中使用的缓冲区
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // 返回概率最大的索引
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // 根据概率分布采样索引（概率需归一化为 1）
    // coin 为 [0,1) 的随机数，通常来自 random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // 作为数值舍入误差的兜底
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top‑p（核）采样：从累积概率达到阈值的最小集合中采样
    // 该集合的累积概率达到 topp，避免采到极低概率的 token
    // 以降低输出“跑偏”的概率
    // coin 为 [0,1) 的随机数，通常来自 random_f32()

    int n0 = 0;
    // 按概率从大到小快速排序下标
    // 小于 (1 - topp)/(n - 1) 的值不可能入选，可提前裁剪
    // 为提升效率，排序前裁掉这些候选
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // 在累积概率超过 topp 处截断列表
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // 作为数值舍入误差的兜底 consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; 
        }
    }

    // 在截断后的列表中采样
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // 作为数值舍入误差的兜底
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // 仅核采样用到的缓冲区；体量较小
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift 伪随机数生成器
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // 生成 [0,1) 的 float32 随机数
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // 根据 logits 与超参数采样下一个 token
    int next;
    if (sampler->temperature == 0.0f) {
        // 贪心采样：直接取最大概率的 token
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // 对 logits 应用温度系数
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // 对 logits 执行 softmax 得到概率分布
        softmax(logits, sampler->vocab_size);
        // 抛“硬币”作为采样的随机源
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // 直接从概率分布采样
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p（核）采样：截断低概率的 token
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// 工具：计时函数

long time_in_ms() {
    // 返回毫秒计时，用于性能评测
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

long long get_current_process_memory() {
    FILE *proc_file;
    unsigned long resident_pages;
    long page_size;

    // 打开/proc/self/statm文件，该文件包含当前进程的内存状态信息
    proc_file = fopen("/proc/self/statm", "r");
    if (proc_file == NULL) return -1;

    // /proc/self/statm格式：总内存页 驻留内存页 共享页 文本页 库页 数据页 脏页
    // 我们只需要第二列：驻留内存页
    if (fscanf(proc_file, "%*u %lu", &resident_pages) != 1) {
        fclose(proc_file);
        return -1;
    }
    fclose(proc_file);

    // 获取系统页大小（字节）
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size == -1) {
        return -1;
    }

    // 计算内存占用（转换为KB）
    return (resident_pages * page_size) / 1024;
}

// ----------------------------------------------------------------------------
// 文本生成循环

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // 将字符串形式的 prompt 编码为 token 序列
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // 额外空间：字符串结尾与可选 BOS/EOS
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // 主循环开始
    long start = time_in_ms(), fTT=0, mem=0;  // 用于计时（在第 1 次迭代后初始化）
    int next;        // 存放下一个 token
    int token = prompt_tokens[0]; // 使用 prompt 的第一个 token 启动
    int pos = 0;     // 序列中的位置
    while (pos < steps) {

        // 前向计算得到下一个 token 的 logits
        float* logits = forward(transformer, token, pos);
        if (pos==0) {
            fTT = time_in_ms()-start;
            mem = get_current_process_memory();
        }
        // 推进生成状态机
        if (pos < num_prompt_tokens - 1) {
            // 若仍在处理输入 prompt，则强制使用下一个 prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // 否则从 logits 采样下一个 token
            next = sample(sampler, logits);
        }
        pos++;

        // 数据依赖的终止条件：BOS(=1) 作为分隔符
        if (next == 1) { break; }

        // 将 token 解码为字符串并打印
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // 等同于 printf，但会跳过不安全字节
        fflush(stdout);
        token = next;

        // 在此初始化计时器，因为首轮可能较慢
        if (pos==1) { start = time_in_ms(); }
    }
    printf("\n");

    // 报告吞吐（token/s）。计时从第 1 次迭代后开始
    if (pos > 1) {
        long end = time_in_ms();
        double tPTok = (double) (end-start) / 1000 / (pos-1);
        fprintf(stderr, "achieved tok/s: %f\n", 1./tPTok);
        fprintf(stderr, "FTL/ms: %ld\n", fTT);
        fprintf(stderr, "Memory usage(Mb): %f\n", (double)(mem)/1024);
        // fprintf(stderr, "Estimated power consumption per 1M tok/ J: %f, / kWh: %f \n", tPTok * 1000000 * 45, tPTok * 1000000 * 45 / 3600000);
        fprintf(stderr, "Estimated power consumption per 1M tok/ kWh: %f \n", tPTok * 1000000 * 45 / 3600000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // 从标准输入读取一行（不含换行符）
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // 去掉换行符
        }
    }
}

// ----------------------------------------------------------------------------
// 聊天循环（简易实现，仅作演示）
// 我手动对比过若干聊天示例与 Python 参考实现的分词结果
// 二者基本一致，但该功能未经过完整测试，
// 实现也未做完备的健壮性处理，目前更偏向原型演示

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {


    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // 主循环开始
    int8_t user_turn = 1;
    int next;        // 存放下一个 token
    int token;      
    int prev_token;
    int pos = 0;     // 序列中的位置
    while (pos < steps) {

        // 当轮到用户向对话贡献 token 时……
        if (user_turn) {
            // 获取位于位置 0 的（可选）系统提示词
            if (pos == 0) {
                // 在位置 0，用户也可以提供一个系统提示词
                if (cli_system_prompt == NULL) {
                    // 未传入系统提示词，尝试从标准输入（stdin）获取
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // 已传入系统提示词，直接使用它
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // 获取用户提示词
            if (pos == 0 && cli_user_prompt != NULL) {
                // 位置 0 的用户提示词已传入，直接使用
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // 否则从标准输入（stdin）获取用户提示词
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // 将用户/系统提示词渲染（转换）为 Llama 2 Chat 的对话模式结构
            if (pos == 0 && system_prompt[0] != '\0') {
                char system_template[] = "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char user_template[] = "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // 将渲染后的提示词编码为 token
            encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            user_idx = 0; 
            user_turn = 0;
            printf("Assistant: ");
        }

        // 确定下一个要传入 Transformer 的 token
        if (user_idx < num_prompt_tokens) {
            // 若仍在处理输入 prompt，则强制使用下一个 prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // 否则使用上一次采样得到的下一个 token
            token = next;
        }
        // EOS（=2）token 表示助手回合结束
        if (token == 2) { user_turn = 1; }

        // 前向计算得到下一个 token 的 logits
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // 助手正在回复，因此打印它的输出
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // 等同于 printf，但会跳过不安全字节
            fflush(stdout);
        }
        if (next == 2) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}


// ----------------------------------------------------------------------------
// 命令行入口（非测试时编译）
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <checkpoint> [options]\n");
    fprintf(stderr, "Example: run model.bin -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // 默认参数
    char *checkpoint_path = NULL; 
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   
    float topp = 0.9f;          // 核采样的 top‑p；1.0 关闭；0.9 效果好但稍慢
    int steps = 256;            // 生成步数
    char *prompt = NULL;        // 提示词字符串
    unsigned long long rng_seed = 0; // 默认使用当前时间作为随机种子
    char *mode = "generate";    // 模式：generate|chat
    char *system_prompt = NULL; // 聊天模式下可选的系统提示词

    // 简易命令行参数解析
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); } 
        if (argv[i][0] != '-') { error_usage(); } 
        if (strlen(argv[i]) != 2) { error_usage(); } 

        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // 参数校验与修正
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // 从模型 .bin 构建 Transformer
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; 

    // 通过 tokenizer.bin 构建分词器
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // 构建采样器
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // 开始运行！
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        // chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // 资源回收与清理
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
