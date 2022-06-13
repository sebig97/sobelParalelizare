
# Paralelizare operator Sobel edge detection

Proiectul isi propune sa compare performanta din punct de vedere al timpului intre calculul GPU si CPU, implementand un algoritm de detectare a marginilor folosind filtrul Sobel.

Proiectul este efectuat in doua parti, cu implementari atat pentru GPU (implementare Cuda), cat si pentru CPU (implementare fara Cuda). Imaginile folosite sunt cu rezolutie diferita, astfel incat timpul de calcul va fi afectat pentru fiecare unitate. 
Implementarea GPU e realizata intr-un kernel, printr-un script in Python folosit ca wrapper.

Analiza si compararea se face pe 5 tipuri de imagini cu rezolutiile urmatoare:
- VGA (640X480)
- HD (1280X720)
- FHD (1920X1080)
- QHD (3840X2160)
- 4K (5120X3200)


Utilizand initial CPU, pentru o imagine de 640X480, timpul total de aplicare al filtrului este de aprox 10.39sec. Pe masura ce imaginea este mai mare, timpul necesar ajunge la 28.67 sec pentru o imagine de 1280x720, respectiv 65.10 sec pentru o imagine de 1920X1080 si 265.71 pentru una de 3840X2160	

Ca orice filtru, vorbim despre o matrice 3x3 care a facut convolutie cu o altă matrice 2D mai mare, anume imaginea. Un exemplu ar fi aplicarea aceluiași filtru la cele 3 canale dintr-o imagine color separat și obținerea a 3 ieșiri. Dacă vreau sa detectez marginile dintr-o imagine color folosind cât mai multe informații posibil, în loc să fac o singură conversie și să aruncați celelalte canale, pot să aleg iesirea din cele trei canale și apoi să adaug mărimile pe toate canalele. Observ ca daca adaug doar gradienții de la diferite canale, este posibil să am două margini puternice cu semne opuse anulându-se, așa că e nevoie de un fel de neliniaritate aici pentru a face asta.

Algoritmul Sobel funcționează prin măsurarea intensității diferite a pixelilor dintr-o imagine. Aacest lucru este cel mai ușor de realizat atunci când imaginea este o imagine standardizată în tonuri de gri, astfel se foloseste functia de gray scale. Există mai mulți algoritmi pentru a genera imagini în tonuri de gri - cel mai simplu fiind o medie a valorilor R, G și B dintr-un pixel. Apoi, se redistribuie aceste noi valori de intensitate câmpurilor R, G și B ale pixelului original. Orice situație în care toate valorile R, G și B sunt aceleași va oferi o nuanță de gri. Cu toate acestea, deși acest lucru dă in cele mai multe cazuri rezultate bune, functia din proiect utilizează o medie ponderată care este mai bine aliniată cu modul în care ochiul uman interpretează diferitele culori și cu tranziția dintre ele:

            imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
            
## Appendix

Am realizat paralelizarea functiei de aplicare a filtrului Sobel si am ajuns la un timp de executie de 0.22 secunde pentru o imagine de 640X480. S-a paralelizat folosind CUDA si comenzi specifice limbajului:
- prin "global" se declara functia
- cele 3 variabile threadIdx, blockDim si blockIdx se utilizeaza pentru a extrage indexul din imaginea actuala (prin index se poate accesa elementele din vectori), precum si destinatia pixelilor in memorie. Tipul lor este dim3, unde blockDim reprezinta dimensiunea blocului, adica nr de threaduri din bloc in directia x,y,.. De asemenea se foloseste pentru a face load la kernel.
- kernelul se apeleaza prin functia de sobel_filter, iar aceasta se configureaza prin parametri (const float * pixin, float * pixout, const int width, const int height) si astfel pot vedea nr de threaduri care se executa pe device in anumite zone.
- cateva functii predefinite cuda: memcpy_htod prin care se poate copia inputului (in bytes) in gpu de la host la device, mem_alloc ce aloca memorie in gpu pentru imaginea de input, memcpy_dtoh ce e operatia inversa htod, adica se copiaza rezultatul de la device la host.



| Implementation | Image size | Convert   | Allocate memory | Image processing | Convert from GPU | Saving    | Total time |
| -------------- | ---------- | ----------| --------------- | ---------------- | -----------------| --------- | ---------- |
| **CPU**        | 640X480    | 0.0200002 | -               | 9.2549998        | -                | 0.1200001 | 10.395002  |
| **GPU**        | 640X480    | 0.0609910 | 0.0066528       | 0.0022110        | 0.0023078        | 0.1253440 | 0.2256432  |
|                |            |           |                 |                  |                  |           |            |
| **CPU**        | 1280X720   | 0.0199999 | -               | 27.481999        | -                | 0.1700000 | 29.671999  |
| **GPU**        | 1280X720   | 0.0299420 | 0.0145668       | 0.0002372        | 0.0056648        | 0.1365141 | 0.1929452  |
|                |            |           |                 |                  |                  |           |            |
| **CPU**        | 1920X1080  | 0.0799999 | -               | 63.271000        | -                | 0.7500000 | 67.101000  |
| **GPU**        | 1920X1080  | 0.0938701 | 0.0312290       | 0.0003209        | 0.0103600        | 0.5749020 | 0.8405869  |
|                |            |           |                 |                  |                  |           |            |
| **CPU**        | 3840X2160  | 0.3199999 | -               | 250.45799        | -                | 2.9400000 | 265.71432  |
| **GPU**        | 3840X2160  | 0.2851300 | 0.0956630       | 0.0002830        | 0.0324380        | 1.9789669 | 2.9856544  |
|                |            |           |                 |                  |                  |           |            |
| **CPU**        | 5120X3200  | 0.5800001 | -               | 532.90599        | -                | 5.3599998 | 555.54334  |
| **GPU**        | 5120X3200  | 0.4501910 | 0.1823840       | 0.0002810        | 0.0589640        | 3.8527259 | 5.5385435  |

Surse de inspiratie:
- la kernelul Cuda am utilizat metode pentru filtrul sobel, kernel sobel prin analiza comentariilor si informatiilor din https://github.com/JakubDziworski/Cuda-Sobel/blob/master/kernel.cu
- algoritmul (formula) de conversie a unei imagini rgb -> grayscale: https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion#:~:text=Average%20method%20is%20the%20most,Its%20done%20in%20this%20way. / https://aryamansharda.medium.com/how-image-edge-detection-works-b759baac01e2
- Functiile de sobel si convolutie: https://stackoverflow.com/questions/46513267/how-to-improve-the-efficiency-of-a-sobel-edge-detector
- functii din kernel, intelegerea algoritm de edge detection, implementarile cpu, gpu: https://webbut.unitbv.ro/index.php/Series_I/article/view/1084/968
