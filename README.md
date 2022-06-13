
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

Am realizat paralelizarea functiei de aplicare a filtrului Sobel si am ajuns la un timp de executie de 0.22 secunde pentru o imagine de 640X480. S-a paralelizat folosind CUDA si comenzi specifice limbajului.


