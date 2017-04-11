# include <cv.h>
# include <highgui.h>
# include <stdlib.h>
# include <stdio.h>

// рассчитать
хеш
картинки
__int64
calcImageHash(IplImage * image, bool
show_results = false);
// рассчёт
расстояния
Хэмминга
__int64
calcHammingDistance(__int64
x, __int64
y);

int
main(int
argc, char * argv[])
{
    IplImage * object = 0, *image = 0;

char
obj_name[] = "cat2.jpg";
char
img_name[] = "cat.jpg";

// имя
объекта
задаётся
первым
параметром
char * object_filename = argc >= 2 ? argv[1]: obj_name;
// имя
картинки
задаётся
вторым
параметром
char * image_filename = argc >= 3 ? argv[2]: img_name;

// получаем
картинку
object = cvLoadImage(object_filename, 1);
image = cvLoadImage(image_filename, 1);

printf("[i] object: %s\n", object_filename);
printf("[i] image: %s\n", image_filename);
if (!object){
    printf("[!] Error: cant load object image: %s\n", object_filename);
return -1;
}
if (!image){
printf("[!] Error: cant load test image: %s\n", image_filename);
return -1;
}

// покажем
изображение
cvNamedWindow("object");
cvShowImage("object", object);
cvNamedWindow("image");
cvShowImage("image", image);

// построим
хэш
__int64
hashO = calcImageHash(object, true);
// cvWaitKey(0);
__int64
hashI = calcImageHash(image, false);

// рассчитаем
расстояние
Хэмминга
__int64
dist = calcHammingDistance(hashO, hashI);

printf("[i] Hamming distance: %I64d \n", dist);

// ждём
нажатия
клавиши
cvWaitKey(0);

// освобождаем
ресурсы
cvReleaseImage( & object);
cvReleaseImage( & image);

// удаляем
окна
cvDestroyAllWindows();
return 0;
}

// рассчитать
хеш
картинки
__int64
calcImageHash(IplImage * src, bool
show_results)
{
if (!src){
return 0;
}

IplImage * res = 0, *gray = 0, *bin = 0;

res = cvCreateImage(cvSize(8, 8), src->depth, src->nChannels);
gray = cvCreateImage(cvSize(8, 8), IPL_DEPTH_8U, 1);
bin = cvCreateImage(cvSize(8, 8), IPL_DEPTH_8U, 1);

// уменьшаем
картинку
cvResize(src, res);
// переводим
в
градации
серого
cvCvtColor(res, gray, CV_BGR2GRAY);
// вычисляем
среднее
CvScalar
average = cvAvg(gray);
printf("[i] average: %.2f \n", average.val[0]);
// получим
бинарное
изображение
относительно
среднего
// для
этого
воспользуемся
пороговым
преобразованием
cvThreshold(gray, bin, average.val[0], 255, CV_THRESH_BINARY);

// построим
хэш
__int64
hash = 0;

int
i = 0;
// пробегаемся
по
всем
пикселям
изображения
for (int y=0; y < bin->height; y++ ) {
uchar * ptr = (uchar *)(bin->imageData + y * bin->widthStep);
for (int x=0; x < bin->width; x++ ) {
// 1 канал
if (ptr[x]){
// hash |= 1 << i; // warning C4334: '<<': result
of
32 - bit
shift
implicitly
converted
to
64
bits(was
64 - bit
shift
intended?)
hash |= 1
i64 << i;
}
i + +;
}
}

printf("[i] hash: %I64X \n", hash);

if (show_results)
{
// увеличенные
картинки
для
отображения
результатов
IplImage * dst3 = cvCreateImage(cvSize(128, 128), IPL_DEPTH_8U, 3);
IplImage * dst1 = cvCreateImage(cvSize(128, 128), IPL_DEPTH_8U, 1);

// показываем
картинки
cvNamedWindow("64");
cvResize(res, dst3, CV_INTER_NN);
cvShowImage("64", dst3);
cvNamedWindow("gray");
cvResize(gray, dst1, CV_INTER_NN);
cvShowImage("gray", dst1);
cvNamedWindow("bin");
cvResize(bin, dst1, CV_INTER_NN);
cvShowImage("bin", dst1);

cvReleaseImage( & dst3);
cvReleaseImage( & dst1);
}

// освобождаем
ресурсы
cvReleaseImage( & res);
cvReleaseImage( & gray);
cvReleaseImage( & bin);

return hash;
}

// рассчёт
расстояния
Хэмминга
между
двумя
хэшами
// http: // en.wikipedia.org / wiki / Hamming_distance
            // http: // ru.wikipedia.org / wiki / Расстояние_Хэмминга
                        //
                        __int64
calcHammingDistance(__int64
x, __int64
y)
{
__int64
dist = 0, val = x ^ y;

// Count
the
number
of
set
bits
while (val)
    {
        ++dist;
    val &= val - 1;
    }

    return dist;
    }