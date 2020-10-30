
#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define DPRT() printf("%s: %d\n", __FUNCTION__, __LINE__);
#define ARR_SIZE(a) ((int)(sizeof(a) / sizeof(a[0])))
typedef float Decimal;
#define DECIMAL_MIN (FLT_MIN)
#define DECIMAL_MAX (FLT_MAX)
#define TypeArgumentMax (32)
#define FORMAT_BUFFER_SIZE (2048)
#define DIV_SAFETY (1e-15)

////////////////////////////////////////
// Error
////////////////////////////////////////
typedef struct {
    int errorCode;
    const char* errorMessage;
} Error;

void setError(Error* self, int code, const char* message) {
    if(self != NULL) {
        self->errorCode = code;
        self->errorMessage = message;
        
    } else {
        // pass
    }
}

void setNormal(Error* self) {
    if(self != NULL) {
        self->errorCode = 0;
        self->errorMessage = "OK";
    } else {
        // pass
    }
}

int isError(Error* self) {
    if(self != NULL && self->errorCode != 0) {
        return 1;
    } else {
        return 0;
    }
}

void errorExit(Error* self) {
    if(self != NULL) {
        puts(self->errorMessage);
        exit(self->errorCode);
    } else {
        puts("unknown error");
        exit(-9);
    }
}

void errorDump(Error* self) {
    if(self != NULL) {
        printf("errorCode=%d\n", self->errorCode);
        printf("errorMessage=%s\n", self->errorMessage);
    } else {
        printf("errorCode=???\n");
        printf("errorMessage=???\n");
    }
}


////////////////////////////////////////
// Memory
////////////////////////////////////////
void* MemoryAlloc(size_t size) {
    void* mem = malloc(size);
    if(mem == NULL) {
        puts("malloc error");
        exit(9);
    }
    return mem;
}

void MemoryFree(void* p) {
    free(p);
}

////////////////////////////////////////
// Format String
////////////////////////////////////////
char* NewString(const char* fmt, ...) {
    char* buffer = (char*)MemoryAlloc(sizeof(char) * strlen(fmt) * 10);
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buffer, fmt, ap);
    va_end(ap);
    return buffer;
}

char* Format(const char* fmt, ...) {
    static char buffer[FORMAT_BUFFER_SIZE];
    va_list ap;
    va_start(ap, fmt);
    vsprintf(buffer, fmt, ap);
    va_end(ap);
    return buffer;
}


////////////////////////////////////////
// Argument
////////////////////////////////////////
typedef struct {
    int length;
    char* opt[TypeArgumentMax];
    char* value[TypeArgumentMax];
} TypeArgument;

void ArgumentParse(Error* err, TypeArgument* self, int argc, char** argv) {
    int i;
    int cmdIdx, optIdx;
    
    cmdIdx = 0;
    optIdx = 0;
    for(i = 0; i < argc; i++) {
        if(argv[i][0] == '-') {
            self->opt[cmdIdx] = argv[i];
            self->value[cmdIdx] = NULL;
            optIdx = cmdIdx;
            cmdIdx += 1;
            if(cmdIdx >= TypeArgumentMax) {
                setError(err, 9, NewString("argument count is over %d", TypeArgumentMax));
                return;
            }
        } else {
            self->value[optIdx] = argv[i];
        }
    }
    self->length = cmdIdx;
    setNormal(err);
}

void ArgumentDump(TypeArgument* self) {
    int i;
    for(i = 0; i < self->length; i++) {
        printf("%s, %s\n", self->opt[i], self->value[i]);
    }
}

////////////////////////////////////////
// Argument Defines
////////////////////////////////////////
typedef int ArgumentType;
#define ArgumentTypeInt (0)
#define ArgumentTypeDecimal (1)
#define ArgumentTypeString (2)
typedef struct {
    ArgumentType varType;
    const char* optionName;
    void* value;
} TypeArgumentDefine;

typedef struct {
    int size;
    TypeArgumentDefine* defines;
} TypeArgumentDefines;

void ArgumentDefineDump(TypeArgumentDefine* self) {
    char buffer[512];
    switch(self->varType) {
        case ArgumentTypeInt:
            sprintf(buffer, "%s, ArgumentTypeInt, %d", self->optionName, *((int*)self->value));
            break;
        case ArgumentTypeDecimal:
            sprintf(buffer, "%s, ArgumentTypeDecimal, %f", self->optionName, *((Decimal*)self->value));
            break;
        case ArgumentTypeString:
            sprintf(buffer, "%s, ArgumentTypeString, %s", self->optionName, *((char**)self->value));
            break;
        default:
            sprintf(buffer, "%s, Unknown, %p", self->optionName, self->value);
            break;
    }
    printf("%s\n", buffer);
}

void ArgumentDefineAllocation(Error* err, TypeArgumentDefines* self, TypeArgument* args) {
    int i;
    int j;
    int exist;
    TypeArgumentDefine def;
    for(i = 0; i < self->size; i++) {
        def = self->defines[i];
        exist = 0;
        for(j = 0; j < args->length; j++) {
            if(strcmp(def.optionName, args->opt[j]) == 0) {
                if(args->value[i] == NULL) {
                    setError(err, 9, Format("kmeans option allocation error, %s value is nothing", def.optionName));
                    return;
                } else {
                    exist = 1;
                    switch(def.varType) {
                        case ArgumentTypeInt:
                            *((int*)def.value) = atoi(args->value[j]);
                            break;
                        case ArgumentTypeDecimal:
                            *((Decimal*)def.value) = atof(args->value[j]);
                            break;
                        case ArgumentTypeString:
                            *((char**)def.value) = args->value[j];
                            break;
                    }
                }
            }
        }
        
        
        if(exist == 0) {
            setError(err, 9, Format("kmeans option allocation error, %s option is nothing", def.optionName));
            return;
        }
    }
    
    setNormal(err);
    
    // debug
    for(i = 0; i < self->size; i++) {
        ArgumentDefineDump(&self->defines[i]);
    }
}

////////////////////////////////////////
// File
////////////////////////////////////////
typedef struct {
    const char* fileName;
    size_t size;
    char* body;
} TypeFile;

void FileRead(Error* err, TypeFile* self, const char* fileName) {
    struct stat file_status;
    FILE* f;
    char* buffer;
    if(stat(fileName, &file_status) < 0) {
        setError(err, 9, Format("%s, stat(%s) error", __FUNCTION__, fileName));
        return;
    }
    self->size = file_status.st_size;
    f = fopen(fileName, "rb");
    if(f == NULL) {
        setError(err, 9, Format("%s fopen(%s) error, %s", __FUNCTION__, fileName));
        return;
    }
    size_t binSize = sizeof(char) * self->size;
    buffer = (char*)MemoryAlloc(binSize);
    if(fread(buffer, self->size, 1, f) != 1) {
        setError(err, 9, Format("%s fread(%s) error", __FUNCTION__, fileName));
        return;
    }
    setNormal(err);
    self->body = buffer;
}

void FileFree(TypeFile* self) {
    MemoryFree(self->body);
}

////////////////////////////////////////
// TSV
////////////////////////////////////////
typedef struct {
    int m;
    int n;
    char** matrix;
} TypeTSV;

void TSVAnalyze(Error* err, TypeTSV* self, char* buffer) {
    char* sepCRLF;
    char* sepTab;
    int firstCount;
    int nCount;
    int mCount;
    firstCount = -1;
    mCount = 0;
    nCount = 1;
    sepCRLF = strtok(buffer, "\r\n");
    while(sepCRLF != NULL) {
        sepTab = strchr(sepCRLF, '\t');
        nCount = 1;
        while(sepTab != NULL) {
            sepTab += 1;
            sepTab = strchr(sepTab, '\t');
            nCount += 1;
        }
        if(firstCount < 0) {
            firstCount = nCount;
        } else if(firstCount != nCount) {
            //printf("column count error, %d != %d", firstCount, nCount);
            setError(err, 9, Format("%s column count error, %d != %d", __FUNCTION__, firstCount, nCount));
            return;
        }
        sepCRLF = strtok(NULL, "\r\n");
        mCount += 1;
    }
    self->m = mCount;
    self->n = nCount;
    setNormal(err);
}

void TSVInfoDump(TypeTSV* self) {
    printf("TSV info dump\n");
    printf("m=%d\n", self->m);
    printf("n=%d\n", self->n);
}

void TSVAllocation(TypeTSV* self, char* buffer) {
    int m;
    int n;
    int idx;
    char** matrix;
    char* tok;
    size_t binSize = sizeof(char*) * self->m * self->n;
    matrix = (char**)MemoryAlloc(binSize);
    
    idx = 0;
    tok = strtok(buffer, "\r\n\t");
    while(tok != NULL) {
        m = idx / self->n;
        n = idx % self->n;
        matrix[m * self->n + n] = tok;
        tok = strtok(NULL, "\r\n\t");
        idx += 1;
    }
    self->matrix = matrix;
}

char* TSVReference(TypeTSV* self, int i, int j) {
    int idx;
    idx = i * self->n + j;
    return self->matrix[idx];
}

void TSVDump(TypeTSV* self) {
    int i;
    int j;
    int idx;
    for(i = 0; i < self->m; i++) {
        for(j = 0; j < self->n; j++) {
            idx = i * self->n + j;
            printf("%s\t", self->matrix[idx]);
        }
        printf("\n");
    }
}


////////////////////////////////////////
// Matrix
////////////////////////////////////////
typedef struct {
    int m;
    int n;
    size_t binSize;
    Decimal* X;
} TypeMatrix;

void MatrixCreate(TypeMatrix* self, int m, int n) {
    self->m = m;
    self->n = n;
    self->binSize = sizeof(Decimal) * m * n;
    self->X = (Decimal*)MemoryAlloc(self->binSize);
}

void MatrixCopy(Error* err, TypeMatrix* dst, TypeMatrix* src) {
    if(dst->binSize != src->binSize) {
        setError(err, 9, NewString("%s size mismatch %lu != %lu", __FUNCTION__, dst->binSize, src->binSize));
        return;
    }
    memcpy(dst->X, src->X, dst->binSize);
}

void MatrixTranspose(TypeMatrix* dst, TypeMatrix* src) {
    int i;
    int j;
    int srcIdx;
    int dstIdx;
    MatrixCreate(dst, src->n, src->m);
    for(i = 0; i < src->m; i++) {
        for(j = 0; j < src->n; j++) {
            srcIdx = src->n * i + j;
            dstIdx = src->m * j + i;
            dst->X[dstIdx] = src->X[srcIdx];
        }
    }
}

void MatrixTransposeCopy(Error* err, TypeMatrix* dst, TypeMatrix* src) {
    int i;
    int j;
    int srcIdx;
    int dstIdx;
    if(src->m != dst->n || src->n != dst->m) {
        setError(err, 9, NewString("size mismatch, src->m=%d, src->n=%d, dst->m=%d, dst->n=%d", src->m, src->n, dst->m, dst->n));
        return;
    }
    for(i = 0; i < src->m; i++) {
        for(j = 0; j < src->n; j++) {
            srcIdx = src->n * i + j;
            dstIdx = src->m * j + i;
            dst->X[dstIdx] = src->X[srcIdx];
        }
    }
    setNormal(err);
}

void MatrixZeros(TypeMatrix* self, int m, int n) {
    MatrixCreate(self, m, n);
    memset(self->X, 0x00, self->binSize);
}

inline void MatrixSetScalar(TypeMatrix* self, int m, int n, Decimal value) {
    self->X[self->n * m + n] = value;
}

void MatrixSetRandom(TypeMatrix* self) {
    int i, j;
    for(i = 0; i < self->m; i++) {
        for(j = 0; j < self->n; j++) {
            MatrixSetScalar(self, i, j, (Decimal)(rand() % 10));
        }
    }
}

void MatrixDump(TypeMatrix* self) {
    int i, j;
    printf("matrix dump\n");
    for(i = 0; i < self->m; i++) {
        for(j = 0; j < self->n; j++) {
            printf("%f, ", self->X[(i * self->n) + j]);
        }
        printf("\n");
    }
}

////////////////////////////////////////
// Array
////////////////////////////////////////
typedef struct {
    int size;
    size_t binSize;
    Decimal* a;
} TypeArray;

void ArrayCreate(TypeArray* self, int size) {
    self->size = size;
    self->binSize = sizeof(Decimal) * self->size;
    self->a = (Decimal*)MemoryAlloc(self->binSize);
}

void ArrayFree(TypeArray* self) {
    MemoryFree(self->a);
}

void ArrayZeros(TypeArray* self, int size) {
    int i;
    ArrayCreate(self, size);
    for(i = 0; i < size; i++) {
        self->a[i] = 0;
    }
}

void ArrayDump(TypeArray* self) {
    int i;
    printf("a dump: size=%d, ", self->size);
    for(i = 0; i < self->size; i++) {
        printf("%f ", self->a[i]);
    }
    printf("\n");
}


////////////////////////////////////////
// IntArray
////////////////////////////////////////
typedef struct {
    int size;
    size_t binSize;
    int* a;
} TypeIntArray;

void IntArrayCreate(TypeIntArray* self, int size) {
    self->size = size;
    self->binSize = sizeof(int) * self->size;
    self->a = (int*)MemoryAlloc(self->binSize);
}

void IntArrayZeros(TypeIntArray* self, int size) {
    IntArrayCreate(self, size);
    memset(self->a, 0x00, self->binSize);
}

void IntArrayFree(TypeIntArray* self) {
    MemoryFree(self->a);
}


void IntArrayDump(TypeIntArray* self) {
    int i;
    printf("a dump: size=%d, ", self->size);
    for(i = 0; i < self->size; i++) {
        printf("%d ", self->a[i]);
    }
    printf("\n");
}



////////////////////////////////////////
// Array Operation
////////////////////////////////////////
Decimal ArrayTail(TypeArray* self) {
    return self->a[self->size - 1];
}

// ソート済み配列を渡す
int ArrayBisectLeft(int size, Decimal* a, Decimal target) {
    int mid;
    int lo = 0;
    int hi = size;
    
    mid = 0;
    while(lo < hi) {
        mid = (lo + hi) >> 1;
        if(a[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}



////////////////////////////////////////
// IntArray Operation
////////////////////////////////////////
void TSVOutputIntArrayToFile(Error* err, TypeIntArray* iarr, const char* fileName) {
    int i;
    FILE* f;
    f = fopen(fileName, "wb");
    if(f == NULL) {
        setError(err, 9, Format("%s fopen(%s) error, %s", __FUNCTION__, fileName));
        return;
    }
    for(i = 0; i < iarr->size; i++) {
        fprintf(f, "%d\n", iarr->a[i]);
    }
}

////////////////////////////////////////
// Array Calcluation
////////////////////////////////////////
int ArrayCalcArgMin(TypeArray* self) {
    int i;
    int idx;
    idx = 0;
    Decimal min = DECIMAL_MAX;
    for(i = 0; i < self->size; i++) {
        if(self->a[i] < min) {
            idx = i;
            min = self->a[i];
        }
    }
    return idx;
}
Decimal ArrayCalcMin(TypeArray* self) {
    int idx;
    idx = ArrayCalcArgMin(self);
    return self->a[idx];
}

Decimal ArrayCalcDistance(int w, Decimal* a, Decimal* b) {
    int i;
    Decimal tmp;
    Decimal res;
    res = 0;
    for(i = 0; i < w; i++) {
        tmp = a[i] - b[i];
        tmp *= tmp;
        res += tmp;
    }
    res = sqrt(res);
    return res;
}

////////////////////////////////////////
// Matrix Operation
////////////////////////////////////////
inline Decimal* MatrixLine(TypeMatrix* matrix, int idx) {
    return &matrix->X[idx * matrix->n];
}

void MatrixCopyLine(Error* err, TypeMatrix* dst, int dstIdx, TypeMatrix* src, int srcIdx) {
    if(dst->n != src->n) {
        setError(err, 9, NewString("%s columns is defferent, dst:%d != src:%d", __FUNCTION__, dst->n, src->n));
        return;
    }
    memcpy(&dst->X[dstIdx * dst->n], &src->X[srcIdx * dst->n], sizeof(Decimal) * dst->n);
}

void MatrixFillZero(TypeMatrix* matrix) {
    memset(matrix->X, 0x00, matrix->binSize);
}

void TSVOutputMatrixToFile(Error* err, TypeMatrix* matrix, const char* fileName) {
    int i;
    int j;
    FILE* f;
    f = fopen(fileName, "wb");
    if(f == NULL) {
        setError(err, 9, Format("%s fopen(%s) error, %s", __FUNCTION__, fileName));
        return;
    }
    for(i = 0; i < matrix->m; i++) {
        for(j = 0; j < matrix->n; j++) {
            fprintf(f, "%f", matrix->X[i * matrix->n + j]);
            if(j < (matrix->n - 1)) {
                fprintf(f, "\t");
            }
        }
        fprintf(f, "\n");
    }
}



////////////////////////////////////////
// Matrix Calculuation
////////////////////////////////////////
void MatrixCalcFree(TypeArray* self) {
    MemoryFree(self->a);
}
void MatrixCalcAverage(TypeArray* dst, TypeMatrix *matrix) {
    int i;
    int j;
    int idx;
    ArrayZeros(dst, matrix->m);
    for(i = 0; i < matrix->m; i++) {
        for(j = 0; j < matrix->n; j++) {
            idx = matrix->n * i + j;
            dst->a[i] += matrix->X[idx];
        }
    }
    for(i = 0; i < matrix->m; i++) {
        dst->a[i] = dst->a[i] / matrix->n;
    }
}

void MatrixCalcVariance(TypeArray* dst, TypeArray* mean, TypeMatrix *matrix, Decimal ddof) {
    int i;
    int j;
    int idx;
    Decimal dif;
    
    ArrayZeros(dst, matrix->m);
    
    for(i = 0; i < matrix->m; i++) {
        for(j = 0; j < matrix->n; j++) {
            idx = matrix->n * i + j;
            dif = matrix->X[idx] - mean->a[i];
            dst->a[i] += dif * dif;
        }
    }
    for(i = 0; i < matrix->m; i++) {
        dst->a[i] = dst->a[i] / (matrix->n + ddof);
    }
}


void MatrixCalcBroadcastSub(TypeMatrix* matrix, TypeArray* rightSide) {
    int i;
    int j;
    int idx;
    for(i = 0; i < matrix->m; i++) {
        for(j = 0; j < matrix->n; j++) {
            idx = i * matrix->n + j;
            matrix->X[idx] -= rightSide->a[i];
        }
    }
}

void MatrixCalcBroadcastDiv(TypeMatrix* matrix, TypeArray* rightSide) {
    int i;
    int j;
    int idx;
    for(i = 0; i < matrix->m; i++) {
        for(j = 0; j < matrix->n; j++) {
            idx = i * matrix->n + j;
            matrix->X[idx] /= rightSide->a[i];
        }
    }
}

////////////////////////////////////////
// Standard Scaler
////////////////////////////////////////
typedef struct {
    int n;
    TypeArray mean;
    TypeArray variance;
    TypeArray std;
} TypeStandardScaler;

void StandardScalerFit(Error* err, TypeStandardScaler* self, TypeMatrix* matrix) {
    int i;
    MatrixCalcAverage(&self->mean, matrix);
    MatrixCalcVariance(&self->variance, &self->mean, matrix, 0);
    ArrayCreate(&self->std, self->mean.size);
    for(i = 0; i < matrix->m; i++) {
        self->variance.a[i] += DIV_SAFETY;
        self->std.a[i] = sqrt(self->variance.a[i]);
    }
}

void StandardScalerTransForm(TypeStandardScaler* self, TypeMatrix* matrix) {
    // (x - u) / std
    
    //ArrayDump(&self->mean);
    //ArrayDump(&self->variance);
    //ArrayDump(&self->std);
    
    MatrixCalcBroadcastSub(matrix, &self->mean);
    MatrixCalcBroadcastDiv(matrix, &self->std);
}

////////////////////////////////////////
// KMeansOption
////////////////////////////////////////
typedef struct {
    char* fileName;
    int clusters;
    int maxIter;
    Decimal tol;
} KMeansOptions;

void KmeansOptionDump(KMeansOptions* self) {
    printf("fileName=%s\n", self->fileName);
    printf("clusters=%d\n", self->clusters);
    printf("maxIter=%d\n", self->maxIter);
    printf("tol=%f\n", self->tol);
}

////////////////////////////////////////
// KMeansPlusplus
////////////////////////////////////////
void KMeansPlusplus(Error* err, TypeMatrix* centroid, int clusters, TypeMatrix* mat) {
    
    srand(time(NULL));
    
    int i;
    int j;
    int k;
    int midx;
    int confirmedCount;
    Decimal *center;
    Decimal *row;
    Decimal distance;
    Decimal maxn;
    Decimal prob;
    TypeArray distances;
    TypeArray minDistances;
    TypeIntArray used;
    
    MatrixCreate(centroid, clusters, mat->n);
    IntArrayZeros(&used, mat->m);
    midx = (rand() % mat->m);
    used.a[midx] = 1;
    confirmedCount = 1;
    MatrixCopyLine(err, centroid, 0, mat, midx);
    if(isError(err)) {
        return;
    }
    ArrayCreate(&minDistances, mat->m);
    while(confirmedCount <= clusters) {
        ArrayCreate(&distances, confirmedCount);
        for(i = 0; i < mat->m; i++) {
            if(used.a[i] != 0) {
                minDistances.a[i] = 0;
                continue;
            }
            for(j = 0; j < confirmedCount; j++) {
                //distances
                center = MatrixLine(mat, j);
                row = MatrixLine(mat, i);
                distance = ArrayCalcDistance(mat->n, center, row);
                distance = distance * distance;
                distances.a[j] = distance;
            }
            minDistances.a[i] = ArrayCalcMin(&distances);
        }
        
        // distance to probability
        for(k = 1; k < mat->m; k++) {
            minDistances.a[k] = minDistances.a[k] + minDistances.a[k - 1];
        }
        
        maxn = ArrayTail(&minDistances);
        for(k = 0; k < mat->m; k++) {
            minDistances.a[k] = minDistances.a[k] / maxn;
        }
        
        
        prob = (Decimal)((double)rand() / RAND_MAX);
        midx = ArrayBisectLeft(minDistances.size, minDistances.a, prob);
        if(midx >= minDistances.size) {
            midx = minDistances.size - 1;
        }
        used.a[midx] = 1;
        
        
        MatrixCopyLine(err, centroid, confirmedCount, mat, midx);
        if(isError(err)) {
            return;
        }
        ArrayFree(&distances);
        
        confirmedCount += 1;
    }
    setNormal(err);
}


////////////////////////////////////////
// KMeans
////////////////////////////////////////
typedef struct {
    int m;
    int n;
    int clusters;
    int maxIter;
    Decimal tol;
} KMeans;

void Test1() {
    Decimal arr[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5};
    int arrSize = sizeof(arr) / sizeof(arr[0]);
    for(int i = 0; i < 7; i++) {
        printf("target=%f, idx=%d\n", (Decimal)i, ArrayBisectLeft(arrSize, arr, i));
    }
}

void E_Step(TypeMatrix* centroid, TypeIntArray* predicts, TypeMatrix* X) {
    int i;
    int j;
    Decimal* center;
    Decimal* row;
    TypeArray distances;
    ArrayCreate(&distances, centroid->m);
    for(i = 0; i < X->m; i++) {
        row = MatrixLine(X, i);
        for(j = 0; j < centroid->m; j++) {
            center = MatrixLine(centroid, j);
            distances.a[j] = ArrayCalcDistance(centroid->n, center, row);
        }
        predicts->a[i] = ArrayCalcArgMin(&distances);
    }
    ArrayFree(&distances);
}

void M_Step(TypeMatrix* new_centroid, TypeIntArray* predicts, TypeMatrix* Xt) {
    int i;
    int j;
    int k;
    
    
    TypeArray counts;
    ArrayCreate(&counts, new_centroid->m);
    for(i = 0; i < predicts->size; i++) {
        for(j = 0; j < new_centroid->m; j++) {
            if(predicts->a[i] == j) {
                counts.a[j] += 1;
            }
        }
    }
    
    MatrixFillZero(new_centroid);
    for(i = 0; i < new_centroid->m; i++) {
        for(j = 0; j < new_centroid->n; j++) {
            for(k = 0; k < predicts->size; k++) {
                if(predicts->a[k] == i) {
                    MatrixLine(new_centroid, i)[j] += MatrixLine(Xt, j)[k];
                }
            }
        }
    }
    
    for(i = 0; i < new_centroid->m; i++) {
        for(j = 0; j < new_centroid->n; j++) {
            MatrixLine(new_centroid, i)[j] /= counts.a[i];
        }
    }
}


int main(int argc, char** argv) {
    int i;
    int j;
    int iteration;
    char* buf;
    Error err;
    TypeMatrix X;
    TypeMatrix Xt;
    Decimal center_shift;
    Decimal center_shift_tot;
    
    ////////////////////////////////////////
    // application arguments
    ////////////////////////////////////////
    TypeArgument args;
    ArgumentParse(&err, &args, argc, argv);
    if(isError(&err) != 0) {
        errorExit(&err);
    }
    //ArgumentDump(&args); // debug
    
    
    
    ////////////////////////////////////////
    // kmeans options
    ////////////////////////////////////////
    KMeansOptions kmeansOptions;
    TypeArgumentDefine definesArray[] = {
        {ArgumentTypeString, "-fileName", (void*)&kmeansOptions.fileName},
        {ArgumentTypeInt, "-clusters", (void*)&kmeansOptions.clusters},
        {ArgumentTypeInt, "-maxIter", (void*)&kmeansOptions.maxIter},
        {ArgumentTypeDecimal, "-tol", (void*)&kmeansOptions.tol},
    };
    TypeArgumentDefines defines = {
        ARR_SIZE(definesArray),
        definesArray
    };
    ArgumentDefineAllocation(&err, &defines, &args);
    if(isError(&err)) {
        errorExit(&err);
    }
    // KmeansOptionDump(&kmeansOptions); // debug
    
    
    
    ////////////////////////////////////////
    // read TSV
    ////////////////////////////////////////
    TypeFile file;
    FileRead(&err, &file, kmeansOptions.fileName);
    if(isError(&err)) {
        errorExit(&err);
    }
    TypeTSV tsv;
    TSVAnalyze(&err, &tsv, file.body);
    if(isError(&err)) {
        errorExit(&err);
    }
    //TSVInfoDump(&tsv); // debug
    FileFree(&file);// 開放しておく
    FileRead(&err, &file, kmeansOptions.fileName);
    if(isError(&err)) {
        errorExit(&err);
    }
    TSVAllocation(&tsv, file.body);
    //TSVDump(&tsv);
    
    
    
    ////////////////////////////////////////
    // setup matrix, matrixT
    ////////////////////////////////////////
    MatrixCreate(&X, tsv.m, tsv.n);
    MatrixTranspose(&Xt, &X);
    for(i = 0; i < tsv.m; i++) {
        for(j = 0; j < tsv.n; j++) {
            buf = TSVReference(&tsv, i, j);
            MatrixSetScalar(&X, i, j, atof(buf));
        }
    }
    
    MatrixTransposeCopy(&err, &Xt, &X);
    if(isError(&err)) {
        errorExit(&err);
    }
    FileFree(&file);// 行列に展開したので不要
    
    ////////////////////////////////////////
    // scaleing
    ////////////////////////////////////////
    TypeStandardScaler scaler;
    StandardScalerFit(&err, &scaler, &Xt);
    if(isError(&err)) {
        errorExit(&err);
    }
    StandardScalerTransForm(&scaler, &Xt);
    MatrixTransposeCopy(&err, &X, &Xt);
    if(isError(&err)) {
        errorExit(&err);
    }
    //MatrixDump(&X);
    
    
    
    ////////////////////////////////////////
    // kmeans init
    ////////////////////////////////////////
    TypeMatrix centroid;
    KMeansPlusplus(&err, &centroid, kmeansOptions.clusters, &X);
    if(isError(&err)) {
        errorExit(&err);
    }
    
    
    
    ////////////////////////////////////////
    // em-step
    ////////////////////////////////////////
    TypeIntArray predicts;
    TypeMatrix new_centroid;
    IntArrayCreate(&predicts, X.m);
    MatrixCreate(&new_centroid, kmeansOptions.clusters, X.n);
    for(iteration = 0; iteration < kmeansOptions.maxIter; iteration++) {
        E_Step(&centroid, &predicts, &X);
        M_Step(&new_centroid, &predicts, &Xt);
        
        center_shift_tot = 0;
        for(i = 0; i < centroid.m; i++) {
            center_shift = ArrayCalcDistance(centroid.n, MatrixLine(&centroid, i), MatrixLine(&new_centroid, i));
            center_shift *= center_shift;
            center_shift_tot += center_shift;
        }
        if(center_shift_tot < kmeansOptions.tol) {
            printf("iteration=%d, center_shift_tot(%f) < tol(%f)\n", iteration, center_shift_tot, kmeansOptions.tol);
            break;
        }
        
        MatrixCopy(&err, &centroid, &new_centroid);
    }
    
    // set predicts
    E_Step(&new_centroid, &predicts, &X);
    //MatrixDump(&new_centroid);
    //IntArrayDump(&predicts);
    
    
    
    ////////////////////////////////////////
    // out put
    ////////////////////////////////////////
    TSVOutputMatrixToFile(&err, &new_centroid, "./kmeans_means.txt");
    if(isError(&err)) {
        errorExit(&err);
    }
    TSVOutputIntArrayToFile(&err, &predicts, "./kmeans_predict.txt");
    if(isError(&err)) {
        errorExit(&err);
    }
    
    return 0;
}
// g++ kmeans.cpp -o kmeans -O2 -Wall
// ./kmeans -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b -a -b 
// ./kmeans -fileName seeds_xdata.txt -clusters 3 -maxIter 300 -tol 1e-5


// a dump: size=7, 14.847524 14.559284 0.870999 5.628532 3.258605 3.700202 5.408071
// a dump: size=7, 8.466356 1.705539 0.000568 0.196315 0.142678 2.260694 0.241563
// a dump: size=7, 2.909710 1.305973 0.023850 0.443085 0.377738 1.503571 0.491501
// mean [14.847523809523816, 14.559285714285718, 0.8709985714285714, 5.628533333333335, 3.258604761904762, 3.7002009523809516, 5.408071428571429]
// var [8.426034820861675, 1.6974066326530617, 0.0005556905217687077, 0.19537045841269837, 0.14198882950113376, 2.2499188835229025, 0.24040282823129258]
// std [2.9027633077572266, 1.3028455904876302, 0.02357308893142152, 0.4420073058363384, 0.37681405162378667, 1.4999729609305972, 0.49030891102578644]
