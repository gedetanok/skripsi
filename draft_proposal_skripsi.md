# PROPOSAL SKRIPSI

**Judul:**
Analisis Komparatif Teknik *Preprocessing* Citra Fundus dan Arsitektur *Deep Learning* untuk Klasifikasi Tingkat Keparahan *Diabetic Retinopathy* pada Berbagai Dataset Publik

---

## PENDAHULUAN

### 1.1 Latar Belakang

*Diabetic retinopathy* (DR), sebuah komplikasi mikrovaskuler progresif dari diabetes melitus, merupakan penyebab utama kebutaan yang dapat dicegah pada kelompok usia produktif (American Diabetes Association, 2024), dan skala persoalannya meluas dengan cepat: Teo et al. (2021) memproyeksikan 103 juta orang dewasa yang terdampak pada tahun 2020 akan mencapai sekitar 161 juta pada tahun 2045. Indonesia menghadapi tren ini secara akut. Insidens DR sebesar 34,6 per 1.000 orang-tahun (Sasongko et al., 2025) dan prevalensi 55% di antara pasien diabetes di sebuah rumah sakit rujukan (Saputra et al., 2024) terjadi bersamaan dengan ketersediaan oftalmolog yang kurang dari dua per 100.000 penduduk dan terkonsentrasi di wilayah urban Pulau Jawa, sehingga pemeriksaan fundus tahunan yang diamanatkan pedoman klinis mustahil dikerjakan secara manual pada skala yang dibutuhkan. Satu-satunya respons yang layak adalah mengotomatisasi interpretasi citra fundus di layanan kesehatan primer, dengan tugas relevan yang bukan sekadar menandai ada atau tidaknya penyakit, melainkan menetapkan tingkat keparahan pada skala lima tingkat *International Clinical DR* (ICDR) (Wilkinson et al., 2003), karena tingkat itulah yang menentukan apakah seorang pasien dirujuk. Kesalahan karena itu bersifat mahal pada kedua arah: tingkat yang terlalu rendah menunda rujukan yang dibutuhkan pasien, sedangkan tingkat yang terlalu tinggi menghabiskan waktu spesialis yang memang sudah langka, sehingga akurasi tepat pada batas-batas antartingkat itulah yang menentukan.

*Deep learning* telah mengubah *grading* DR otomatis dari sekadar aspirasi menjadi kenyataan yang berfungsi sepanjang sepuluh tahun terakhir. Dengan *convolutional neural network* yang dilatih pada 128.175 citra fundus, Gulshan et al. (2016) melaporkan sensitivitas dan spesifisitas di atas 90% untuk *referable DR*; Abramoff et al. (2018) selanjutnya memperoleh izin dari FDA untuk IDx-DR, sistem DR otonom pertama; dan *foundation model* seperti RETFound (Zhou et al., 2023) sejak itu menutup sebagian besar jarak yang tersisa. Dengan mensurvei lebih dari lima puluh studi dan dua puluh dataset, Chopra et al. (2025) menegaskan kematangan ini tetapi mengamati bahwa tantangan utamanya telah bergeser menjauh dari kapabilitas mentah menuju validasi *multi-center* dan kepercayaan klinis. Dengan kata lain, pertanyaan yang belum terjawab bukan lagi apakah sebuah model mampu mengklasifikasikan DR, melainkan apakah angka-angka yang diperoleh pada data yang terkurasi rapi mampu bertahan pada kondisi yang jauh lebih tidak beraturan yang dijumpai di lapangan.

Celah pertama yang terabaikan terletak pada kualitas citra dan penanganannya. Pencahayaan yang tidak konsisten, kontras yang lemah, dan *noise* akuisisi menggerus akurasi *grading* secara sistematis (Anupama et al., 2025), sebuah efek yang paling parah pada kamera berbiaya rendah yang lazim dipakai di layanan kesehatan primer Indonesia; karena penurunan kualitas semacam itu menyembunyikan lesi awal seperti mikroaneurisma, hal itu mendorong model ke arah tingkat yang lebih rendah dari kondisi pasien sesungguhnya. Literatur menawarkan banyak metode *preprocessing* untuk membuat lesi lebih terlihat, mulai dari CLAHE, Ben Graham *preprocessing*, dan ekstraksi *green channel*, hingga usulan yang lebih baru seperti *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*, tetapi metode-metode itu diadopsi secara *ad-hoc*, dengan setiap studi berkomitmen pada satu metode dan tidak pernah menimbangnya terhadap metode lain pada arsitektur dan protokol yang sama. Anupama et al. (2025) mencatat persoalan yang sama dan meminta adanya studi sistematis atas pilihan *preprocessing*, sebab tanpa perbandingan yang setara keputusan itu bertumpu pada dugaan alih-alih bukti.

Celah kedua menyangkut sejauh mana perbandingan-perbandingan tersebut dapat digeneralisasi. Hampir setiap perbandingan *preprocessing* untuk *grading* DR, termasuk studi acuan Anupama et al. (2025), dikonfirmasi hanya pada satu dataset, sehingga teknik atau *backbone* mana pun yang unggul di situ boleh jadi mencerminkan kamera, populasi, dan kualitas khas dataset tersebut alih-alih sifat yang dapat berpindah. Karena citra fundus berbeda secara mencolok dalam hal pencahayaan, warna, dan protokol akuisisi dari satu institusi ke institusi lain, sebuah metode yang unggul pada satu dataset belum tentu unggul pada dataset lain, dan rekomendasi yang dibangun di atas satu dataset merupakan landasan yang rapuh bagi sistem skrining yang dimaksudkan untuk menghadapi kondisi lapangan yang sangat bervariasi. Anupama et al. (2025) karena itu menyarankan pengujian temuan pada beberapa dataset yang mencakup perangkat dan demografi yang beragam; sampai hal itu dilakukan, validitas eksternal dari perbandingan semacam ini tetap belum terbukti.

Dengan mengambil agenda *future work* Anupama et al. (2025) sebagai titik awalnya, penelitian ini menggarap kedua celah tersebut secara bersamaan. Lima teknik *preprocessing* dipasangkan dengan dua *backbone* (ResNet-50 dan ViT-B/16) untuk menghasilkan sepuluh konfigurasi, yang dinilai secara terpisah pada enam dataset publik *grading* DR yang dibangun di atas ontologi ICDR lima tingkat yang sama (IDRiD, DDR, APTOS 2019, Messidor-2, EyePACS, dan DeepDRiD); dalam setiap dataset, setiap konfigurasi dilatih dan diuji pada partisi dataset itu sendiri di bawah satu protokol yang seragam dan dinilai dengan akurasi, *quadratic-weighted kappa* (QWK), dan *macro-F1*. Untuk menemukan konfigurasi yang unggul bukan pada satu dataset melainkan secara seragam pada keenam dataset, nilai QWK per dataset digabungkan melalui uji Friedman (Demsar, 2006), dan konfigurasi-konfigurasi terdepan dibandingkan dengan uji *Wilcoxon signed-rank* antardataset serta dengan uji McNemar per dataset, sementara Grad-CAM memeriksa dasar sesungguhnya dari prediksi konfigurasi terdepan. Kontribusi penelitian ini ada tiga: (i) perbandingan yang *reproducible* dan setara antara metode *preprocessing* dan *backbone* untuk *grading* DR di bawah satu protokol yang seragam; (ii) pengujian perbandingan tersebut pada enam dataset yang beragam untuk menentukan apakah konfigurasi terbaik bersifat stabil alih-alih terikat pada satu dataset tertentu; dan (iii) panduan berbasis bukti untuk membangun skrining DR otomatis pada layanan kesehatan primer dengan sumber daya terbatas.

### 1.2 Identifikasi Masalah

Berdasarkan latar belakang yang diuraikan pada Bagian 1.1, masalah penelitian dapat diidentifikasi sebagai berikut.

1. Beban *diabetic retinopathy* di Indonesia tinggi dan terus meningkat (insidens 34,6 per 1.000 orang-tahun; Sasongko et al., 2025), sementara rasio oftalmolog kurang dari dua spesialis per 100.000 penduduk dan terkonsentrasi di wilayah urban Pulau Jawa, sehingga skrining fundus tahunan secara manual tidak layak secara struktural dan menuntut sistem interpretasi citra fundus otomatis pada tingkat layanan kesehatan primer.

2. Kualitas citra dari kamera berbiaya rendah yang digunakan di layanan primer sangat tidak konsisten (pencahayaan tidak merata, kontras lemah, dan *noise* akuisisi), dan penurunan kualitas ini menurunkan akurasi *grading* otomatis serta cenderung membuat model menetapkan tingkat di bawah kondisi pasien sesungguhnya, yaitu jenis kesalahan yang menunda rujukan.

3. Cara pemilihan teknik *preprocessing* dalam literatur *grading* DR masih bersifat *ad-hoc*: sebuah studi biasanya memilih satu teknik tanpa mengukurnya terhadap alternatif lain pada arsitektur dan protokol pelatihan yang sama, sehingga bukti setara untuk menginformasikan pilihan tersebut masih belum tersedia (Anupama et al., 2025).

4. Perbandingan *preprocessing* untuk *grading* DR yang ada, termasuk studi acuan Anupama et al. (2025), divalidasi pada satu dataset saja, sehingga belum diketahui apakah *preprocessing* dan *backbone* yang teridentifikasi terbaik pada satu dataset tetap terbaik pada dataset lain dengan kamera, populasi, dan profil kualitas yang berbeda, dan karenanya apakah kesimpulan semacam itu memiliki validitas eksternal alih-alih menjadi artefak dari satu dataset.

5. Belum tersedia panduan berbasis bukti mengenai kombinasi teknik *preprocessing* dan arsitektur *backbone* yang berkinerja konsisten terbaik pada beragam dataset untuk pengembangan sistem skrining DR otomatis bagi layanan kesehatan primer dengan sumber daya terbatas di Indonesia.

### 1.3 Rumusan Masalah

Berdasarkan identifikasi masalah pada Bagian 1.2, penelitian ini dirumuskan melalui tiga rumusan masalah berikut.

1. Pada masing-masing dari keenam dataset, bagaimana lima teknik *preprocessing* (CLAHE, Ben Graham *preprocessing*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*) dan dua arsitektur *backbone* (ResNet-50 dan *Vision Transformer*) memengaruhi kinerja klasifikasi tingkat keparahan *diabetic retinopathy* yang diukur dengan akurasi, *quadratic-weighted kappa* (QWK), dan *macro-F1*?

2. Ketika hasil per dataset diagregasi pada keenam dataset menggunakan uji Friedman, didukung uji *Wilcoxon signed-rank* dan uji McNemar per dataset pada konfigurasi-konfigurasi terdepan, konfigurasi *preprocessing* dan *backbone* mana yang menempati peringkat konsisten tertinggi, dan seberapa stabil peringkat tersebut antardataset?

3. Bagaimana *trade-off* antara kinerja prediktif dan biaya komputasi (jumlah parameter dan waktu inferensi) untuk setiap kombinasi *preprocessing* dan *backbone*, dan konfigurasi mana yang paling sesuai untuk menerapkan sistem skrining DR pada layanan kesehatan primer di Indonesia?

### 1.4 Pembatasan Masalah

Agar penelitian tetap terfokus dan layak diselesaikan dalam rentang waktu penyusunan skripsi, serta untuk mengisolasi pengaruh *preprocessing* dan *backbone* dari faktor lain, masalah yang teridentifikasi pada Bagian 1.2 dibatasi sebagai berikut.

1. Tugas klasifikasi dibatasi pada *grading* lima kelas tingkat keparahan *diabetic retinopathy* pada skala ICDR di tingkat citra, tanpa segmentasi atau deteksi lesi.

2. Dataset yang digunakan adalah enam dataset publik *grading* DR yang berbagi ontologi label ICDR lima kelas, yaitu IDRiD (Porwal et al., 2020), DDR (Li et al., 2019), APTOS 2019 (APTOS, 2019), Messidor-2 (Decenciere et al., 2014), EyePACS (Kaggle dan EyePACS, 2015), dan DeepDRiD (Liu et al., 2022). Setiap dataset diperlakukan sebagai *benchmark* independen tempat semua konfigurasi dilatih dan diuji menggunakan partisi dataset itu sendiri; penelitian ini tidak melakukan transfer lintas-dataset atau adaptasi domain.

3. Teknik *preprocessing* yang dibandingkan dibatasi pada lima metode, yaitu CLAHE, Ben Graham *preprocessing*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*.

4. Arsitektur *backbone* yang dibandingkan dibatasi pada ResNet-50 (CNN) dan ViT-B/16 (*Vision Transformer*), keduanya dilatih melalui *transfer learning* dari bobot ImageNet tanpa *pretraining* khusus domain.

5. Evaluasi bersifat kuantitatif (akurasi, *quadratic-weighted kappa*, *macro-F1*) per dataset, diagregasi pada keenam dataset dengan uji Friedman, dilengkapi uji *Wilcoxon signed-rank* antardataset dan uji McNemar per dataset (dengan koreksi Holm-Bonferroni) pada konfigurasi-konfigurasi terdepan, serta visualisasi Grad-CAM pada konfigurasi terbaik. Studi pembaca (*reader study*) dan *deployment* prospektif berada di luar cakupan.

### 1.5 Tujuan Penelitian

Sejalan dengan rumusan masalah pada Bagian 1.3, penelitian ini memiliki tiga tujuan berikut.

1. Menganalisis pengaruh lima teknik *preprocessing* (CLAHE, Ben Graham *preprocessing*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*) dan dua arsitektur *backbone* (ResNet-50 dan *Vision Transformer*) terhadap kinerja klasifikasi tingkat keparahan *diabetic retinopathy* pada masing-masing dari keenam dataset, yang diukur melalui akurasi, *quadratic-weighted kappa* (QWK), dan *macro-F1*.

2. Mengagregasi hasil per dataset pada keenam dataset menggunakan uji Friedman, didukung uji *Wilcoxon signed-rank* dan uji McNemar per dataset pada konfigurasi-konfigurasi terdepan, untuk mengidentifikasi konfigurasi *preprocessing* dan *backbone* yang menempati peringkat konsisten tertinggi antardataset.

3. Menganalisis *trade-off* antara kinerja prediktif dan biaya komputasi (jumlah parameter dan waktu inferensi) untuk merekomendasikan konfigurasi yang paling sesuai bagi penerapan sistem skrining *diabetic retinopathy* pada layanan kesehatan primer di Indonesia.

### 1.6 Manfaat Penelitian

Dari sisi teoretis, penelitian ini menghasilkan perbandingan empiris yang *reproducible* antara teknik *preprocessing* dan arsitektur *backbone* untuk *grading* DR di bawah satu protokol yang seragam, yang diperkuat dengan menjalankan perbandingan tersebut pada enam dataset publik yang berbagi satu ontologi label namun berbeda dalam perangkat, populasi, dan pencahayaan. Menggabungkan hasil per dataset melalui uji Friedman, didukung uji *Wilcoxon signed-rank* dan uji McNemar per dataset pada konfigurasi-konfigurasi terdepan, memungkinkan penelitian ini menentukan apakah konfigurasi terdepan bertahan antardataset atau sekadar mencerminkan salah satunya, hal yang berbicara langsung kepada agenda *future work* Anupama et al. (2025).

Dari sisi praktis, penelitian ini menyediakan panduan berbasis bukti untuk membangun skrining DR otomatis pada layanan kesehatan primer di Indonesia, sebuah *setting* yang perangkat dan populasinya sangat bervariasi di lapangan. Pasangan *preprocessing* dan *backbone* yang terbukti konsisten paling kuat pada beragam dataset dapat menjadi titik awal *default* bagi pengembangan. Penelitian ini juga menyampaikan *pipeline* eksperimen terdokumentasi yang dapat digunakan kembali untuk mengevaluasi teknik *preprocessing*, *backbone*, atau dataset lain tanpa membangunnya dari awal.

## KAJIAN TEORI

### 2.1 Diabetic Retinopathy

#### 2.1.1 Patofisiologi Singkat

*Diabetic retinopathy* adalah konsekuensi mikrovaskuler kronis dari diabetes melitus. Hiperglikemia yang berkepanjangan melukai kapiler retina, yang menjadi lebih permeabel (menimbulkan edema dan eksudasi lipid) sekaligus mengalami oklusi (menimbulkan iskemia); iskemia itu pada gilirannya menstimulasi *vascular endothelial growth factor* (VEGF) dan pertumbuhan pembuluh darah baru (Wong dan Sabanayagam, 2023). Penyakit ini terbagi secara klinis ke dalam dua tahap. Tahap non-proliferatif (NPDR) menampilkan mikroaneurisma, eksudat keras, *cotton-wool spots*, perdarahan intraretina, *venous beading*, dan *intraretinal microvascular abnormalities* (IRMA) tetapi tanpa pembuluh darah baru, sedangkan tahap proliferatif (PDR) ditandai oleh neovaskularisasi, yang dapat berkembang menjadi perdarahan vitreous dan ablasio retina traksional. Lesi apa yang muncul, dan di mana letaknya, itulah yang mendasari skala keparahan ICDR (Bagian 2.1.3) dan itu pula yang secara persis menjadi sasaran pemeriksaan Grad-CAM (Bagian 2.5.3).

#### 2.1.2 Beban Penyakit Global dan di Indonesia

Sebagaimana dirinci pada Bagian 1.1, DR adalah penyebab utama kebutaan yang dapat dicegah yang beban globalnya diproyeksikan tumbuh dari 103 juta orang pada 2020 menjadi sekitar 161 juta pada 2045 (Teo et al., 2021), sebuah lintasan yang oleh Wong dan Sabanayagam (2023) disebut "pandemi DR". Di Indonesia beban ini diperberat oleh kelangkaan oftalmolog yang parah, kurang dari dua per 100.000 penduduk dan terkonsentrasi di wilayah urban Pulau Jawa, yang menjadi alasan struktural mengapa skrining DR otomatis pada tingkat layanan kesehatan primer dibutuhkan (Sasongko et al., 2025; Saputra et al., 2024; American Diabetes Association, 2024).

#### 2.1.3 Skala International Clinical Diabetic Retinopathy (ICDR)

Variabel luaran klinis dalam penelitian ini adalah skala *International Clinical Diabetic Retinopathy* (ICDR), yang diperkenalkan Wilkinson et al. (2003) sebagai bentuk sederhana dari skala ETDRS. Skala ini memilah retinopati ke dalam lima tingkat berurut: tingkat 0 untuk tidak ada retinopati; tingkat 1 untuk NPDR ringan dengan hanya mikroaneurisma; tingkat 2 untuk NPDR sedang, yang berada di antara ringan dan berat; tingkat 3 untuk NPDR berat, yang didefinisikan oleh aturan "4-2-1" (perdarahan intraretina di keempat kuadran, atau *venous beading* di dua kuadran atau lebih, atau IRMA yang menonjol, tanpa neovaskularisasi); dan tingkat 4 untuk PDR, yang ditandai oleh neovaskularisasi atau perdarahan vitreous/pra-retina.

Skala ini bersifat ordinal: kelima kelasnya berjajar di sepanjang satu sumbu keparahan yang meningkat, sehingga kesalahan yang membentang dua tingkat lebih merugikan daripada yang membentang satu tingkat. Sifat berurut itu menuntut metrik yang peka terhadap urutan kelas, seperti *quadratic-weighted kappa* (Bagian 2.5.2), alih-alih akurasi semata. Biaya kesalahan juga asimetris: memprediksi tingkat 1 untuk pasien yang sesungguhnya tingkat 3 menolak rujukan yang seharusnya diberikan, sedangkan memprediksi tingkat 1 untuk pasien yang sesungguhnya tingkat 0 mengikat kapasitas spesialis tanpa alasan.

### 2.2 Citra Fundus dan Benchmark Dataset

#### 2.2.1 Fotografi Fundus Berwarna

Karena bersifat non-invasif, murah, dan tersedia luas, fotografi fundus berwarna adalah modalitas utama untuk skrining DR; bentuk non-midriatik, yang diambil tanpa mendilatasi pupil, lebih disukai demi kenyamanan pasien dan kemudahan penggunaan. Seberapa dapat dinilainya (*gradable*) sebuah citra bergantung pada katarak, ukuran pupil, dan seberapa merata pencahayaannya, dan sebagian citra terbukti tidak dapat dinilai (*ungradable*), yang menjadi alasan dataset skrining terkini menyertakan label kualitas agar citra semacam itu dapat disaring keluar.

Penelitian ini menggunakan enam dataset publik *grading* DR, masing-masing diperlakukan sebagai *benchmark* independen dan dijelaskan pada Bagian 2.2.2 hingga 2.2.7. Dataset-dataset itu dipilih karena berbagi ontologi label ICDR lima kelas yang sama namun berbeda dalam negara, perangkat akuisisi, populasi, dan kualitas citra; rasionalisasi pemilihannya diberikan pada Bagian 2.2.8, dan rincian partisinya pada Bagian 3.2.

#### 2.2.2 Indian Diabetic Retinopathy Image Dataset (IDRiD)

Sebagai *benchmark* resmi untuk tantangan IEEE ISBI 2018, *Indian Diabetic Retinopathy Image Dataset* (IDRiD; Porwal et al., 2020) menghimpun 516 citra fundus dari satu klinik di Nanded, India, yang diambil dengan satu model kamera (Kowa VX-10α). Hanya subset *Disease Grading* yang digunakan di sini, tempat setiap citra membawa tingkat ICDR dari 0 hingga 4, dan partisi latih dan uji resmi dipertahankan tanpa perubahan agar hasilnya selaras dengan *benchmark* lain.

#### 2.2.3 Dataset for Diabetic Retinopathy (DDR)

*Dataset for Diabetic Retinopathy* (DDR; Li et al., 2019) menyusun 13.673 citra yang diambil dari 147 rumah sakit di 23 provinsi Tiongkok menggunakan beragam jenis kamera, sehingga memberinya variasi perangkat, kualitas, dan demografi yang jauh lebih besar daripada IDRiD. Label mengikuti skala ICDR dari 0 hingga 4 dan disertai label kualitas yang memisahkan citra *gradable* dari *ungradable*. Setelah citra *ungradable* disaring keluar, penelitian ini bekerja dari partisi latih dan uji resmi DDR.

#### 2.2.4 APTOS 2019 Blindness Detection Dataset

Dirilis oleh *Asia Pacific Tele-Ophthalmology Society* bersama Aravind Eye Hospital di India, dataset APTOS 2019 (APTOS, 2019) memuat 3.662 citra fundus berlabel yang diambil di berbagai wilayah pedesaan India dengan beberapa kamera pada kondisi yang bervariasi. *Grading* mengikuti skala ICDR lima kelas; karena kompetisi menahan label pengujiannya, hanya *training set* yang berlabel publik yang digunakan.

#### 2.2.5 Messidor-2

Diakuisisi di Prancis, Messidor-2 (Decenciere et al., 2014) terdiri atas 1.748 citra fundus berpusat pada makula dari 874 pemeriksaan. Tingkat ICDR hasil adjudikasi pihak ketiga yang dirilis publik, yang ditetapkan oleh panel spesialis retina, mencakup 1.744 citra *gradable* dan itulah yang digunakan di sini. Messidor-2 menyumbang populasi Eropa dan perangkat akuisisi yang berbeda dari dataset-dataset Asia.

#### 2.2.6 EyePACS (Kaggle Diabetic Retinopathy Detection)

Sebagai sumber daya DR publik terbesar, dataset EyePACS (Kaggle dan EyePACS, 2015) menghimpun 88.702 citra fundus yang diambil di Amerika Serikat dengan banyak jenis kamera pada kondisi yang sangat bervariasi dan dinilai pada skala ICDR lima kelas. Mengingat sekitar seperempat citranya *ungradable* dan koleksinya sangat besar, penelitian ini mengambil subset yang telah disaring kualitasnya dan distratifikasi per kelas dari partisi resminya, sebagaimana dirinci pada Bagian 3.2.

#### 2.2.7 DeepDRiD

Dataset tantangan ISBI 2020, DeepDRiD (Liu et al., 2022), menyediakan 2.000 citra fundus reguler dari 500 pasien Tiongkok, masing-masing diberi tingkat ICDR lima kelas melalui adjudikasi di antara beberapa oftalmolog. Penelitian ini hanya mempertahankan citra reguler (non *ultra-widefield*) dan, untuk kasus tampilan ganda, menyimpan satu bidang per mata agar setiap masukan berpadanan dengan satu tingkat di tingkat citra.

#### 2.2.8 Rasionalisasi Pemilihan Enam Dataset

Tiga pertimbangan mendasari pemilihan keenam dataset ini. Pertama, semuanya menggunakan satu dan ontologi label yang sama (skala ICDR lima kelas; Wilkinson et al., 2003), yang menjaga kinerja tetap dapat dibandingkan antardataset dan memungkinkan setiap perubahan pada peringkat konfigurasi ditelusuri ke datanya sendiri alih-alih ke anotasi yang tidak konsisten. Kedua, dataset-dataset itu mencakup dimensi yang penting bagi *deployment*, yaitu negara (India, Tiongkok, Prancis, Amerika Serikat), perangkat akuisisi (dari satu kamera di satu lokasi hingga banyak kamera di 147 lokasi), populasi, dan profil kualitas citra, sehingga sebuah konfigurasi yang secara konsisten berkinerja baik pada keenamnya bertumpu pada validitas eksternal yang kokoh alih-alih pada keunikan satu dataset. Ketiga, setiap dataset bersifat publik dan terdokumentasi (Porwal et al., 2020; Li et al., 2019; APTOS, 2019; Decenciere et al., 2014; Kaggle dan EyePACS, 2015; Liu et al., 2022), memenuhi kebutuhan *reproducibility*. Di atas segalanya, setiap dataset berperan sebagai *benchmark* mandiri tempat setiap konfigurasi dilatih sekaligus diuji, sehingga perbandingannya tidak pernah terkontaminasi oleh pergeseran distribusi yang akan ditimbulkan oleh pemindahan model dari satu dataset ke dataset lain.

### 2.3 Teknik Preprocessing untuk Citra Fundus

Citra fundus mentah berbeda kualitasnya karena pencahayaan, respons warna sensor, dan kondisi optik mata semuanya bervariasi. Tujuan *preprocessing* adalah menyeragamkan karakteristik citra ke dalam bentuk yang sama agar lesi klinis (mikroaneurisma, perdarahan, eksudat keras) tertangkap lebih konsisten oleh model. Lima teknik yang dinilai di bawah ini mencakup rentang pendekatan yang ditemukan dalam literatur *grading* DR, dan Gambar 2.1 mengilustrasikan efek visualnya pada sebuah citra fundus.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gambar/prep_demo.png}
\caption{Lima teknik \emph{preprocessing} yang diterapkan pada satu citra fundus berwarna representatif. CLAHE dan LAB-ACE menguatkan kontras lokal, Ben Graham Normalization menstabilkan pencahayaan global, \emph{Adaptive Sigmoid Enhancement} meregangkan rentang intensitas, dan MCIE menggabungkan hasil \emph{green channel}, CLAHE, dan Ben Graham menjadi satu masukan tiga kanal.}
\label{fig:prep}
\end{figure}

#### 2.3.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)

*Contrast Limited Adaptive Histogram Equalization* (CLAHE; Zuiderveld, 1994) menjalankan ekualisasi histogram petak demi petak (*tile*). Di dalam sebuah petak, nilai piksel $r$ dipetakan melalui distribusi kumulatif berbatas kontras $T(r) = (L-1)\sum_{j=0}^{r}\hat{p}(j)$, dengan $L$ menghitung banyaknya tingkat intensitas dan $\hat{p}$ adalah histogram petak setelah dipangkas pada suatu *clip limit* dan kelebihannya didistribusikan ulang; *clip limit* itulah yang menahan penguatan *noise*. Untuk citra fundus, CLAHE dijalankan pada kanal luminans (LAB) agar lesi samar seperti mikroaneurisma, yang jika tidak akan hilang di bawah pencahayaan tak merata, menjadi terlihat.

#### 2.3.2 Ben Graham Normalization

Ben Graham Normalization (Graham, 2015), yang diperkenalkan oleh pemenang kompetisi *Kaggle Diabetic Retinopathy Detection* dan sejak itu diadopsi sebagai langkah rutin *grading* DR, menghilangkan pencahayaan kasar dengan mengurangi versi citra yang dikaburkan Gaussian: $I_{\mathrm{norm}} = \alpha I + \beta G_{\sigma}(I) + \gamma$, dengan $G_{\sigma}$ menyatakan filter Gaussian berjari-jari $\sigma$. Struktur halus seperti lesi dan pembuluh darah menonjol lebih jelas, sementara perbedaan pencahayaan antarkamera diredam.

#### 2.3.3 Adaptive Sigmoid Enhancement

*Adaptive Sigmoid Enhancement* (Anupama et al., 2025) melewatkan intensitas melalui sebuah sigmoid $f(x) = 1 / (1 + \exp(-\alpha (x - \beta)))$, yang kemiringannya $\alpha$ dan titik tengahnya $\beta$ diturunkan secara adaptif dari rata-rata dan simpangan baku intensitas lokal. Transformasi ini menaikkan kontras pada area gelap sembari membiarkan *noise* pada area terang tak tersentuh, sehingga lesi yang pucat menonjol lebih tegas terhadap latar retina.

#### 2.3.4 LAB Adaptive Contrast Enhancement (LAB-ACE)

*LAB Adaptive Contrast Enhancement* (LAB-ACE; Anupama et al., 2025) memindahkan citra ke ruang LAB dan hanya memproses kanal luminans L (CLAHE bersama normalisasi lokal), membiarkan kanal warna A dan B apa adanya, lalu membangun kembali citra RGB. Bekerja pada L saja menaikkan kontras lesi namun menghindari pergeseran warna yang sebaliknya dapat menyesatkan *grading*.

#### 2.3.5 Multi-channel Image Enhancement

*Multi-channel Image Enhancement* (MCIE; Anupama et al., 2025) memadukan beberapa representasi komplementer ke dalam satu masukan tiga kanal, misalnya *green channel* (responsif terhadap hemoglobin), CLAHE yang diterapkan pada kanal L, dan keluaran Ben Graham Normalization. *Backbone* kemudian memanfaatkan beberapa *enhancement* sekaligus alih-alih satu jenis saja; Anupama et al. (2025) mengajukan kombinasi semacam ini sebagai arah masa depan yang menjanjikan.

### 2.4 Arsitektur Deep Learning untuk Klasifikasi Citra

Penelitian ini mengevaluasi dua *backbone* *deep learning* yang mewakili dua paradigma berbeda, yaitu *convolutional neural network* (CNN) yang diwakili ResNet-50 dan *Vision Transformer* (ViT) yang diwakili ViT-B/16. Masing-masing arsitektur dijelaskan pada Bagian 2.4.1 dan 2.4.2.

#### 2.4.1 Convolutional Neural Network dan ResNet-50

Sebuah *convolutional neural network* (CNN) menyisipkan konvolusi, aktivasi non-linear ($\mathrm{ReLU}(z) = \max(0, z)$), dan *pooling* secara berselang. Berbagi bobot menjaga jumlah parameter tetap independen terhadap ukuran citra dan membuat model bersifat ekuivarian-translasi, sementara kedalaman yang lebih besar merakit *receptive field* yang tumbuh dari tepi dan tekstur hingga ke struktur utuh. Fitur akhir menjadi *logit* $z \in \mathbb{R}^{K}$, yang diubah *softmax* $p_k = \exp(z_k) / \sum_{j=1}^{K} \exp(z_j)$ menjadi peluang kelas, dengan keseluruhannya dilatih menggunakan *categorical cross-entropy* $\mathcal{L}_{\mathrm{CE}} = -\sum_{k=1}^{K} y_k \log p_k$.

ResNet-50 (He et al., 2016) termasuk salah satu CNN yang paling banyak digunakan dalam pencitraan medis. Gagasan utamanya adalah koneksi residual $y = F(x) + x$, sebuah jalur pintas yang melawan *vanishing gradient* sehingga membuat jaringan yang sangat dalam dapat dilatih. Jaringan ini menata 50 lapis konvolusi, yang unit berulangnya adalah blok *bottleneck* tiga lapis yang membawa koneksi residual, ke dalam empat tahap dengan resolusi spasial yang menurun secara bertahap, sebagaimana ditunjukkan Gambar 2.2. ResNet-50 diambil sebagai *backbone* CNN di sini karena telah mapan, memiliki jumlah parameter yang moderat (~25,5 juta), dan berulang kali muncul sebagai *baseline* di sepanjang literatur *grading* DR.

\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
  font=\footnotesize,
  box/.style={draw, rounded corners=2pt, minimum height=1.9cm, text width=2.0cm, align=center, font=\scriptsize, inner sep=2pt},
  arr/.style={-{Stealth[length=2mm]}, semithick},
  dim/.style={font=\tiny, text=gray}
]
\node[box, fill=gray!12] (in) {Input fundus\\$224{\times}224{\times}3$};
\node[box, fill=cyan!12, right=0.4cm of in] (c1) {conv1\\$7{\times}7$, 64, /2};
\node[box, fill=gray!8, right=0.4cm of c1] (mp) {max pool\\$3{\times}3$, /2};
\node[box, fill=orange!12, right=0.4cm of mp] (c2) {conv2\_x\\$\left[\begin{smallmatrix}1{\times}1,\,64\\[1pt]3{\times}3,\,64\\[1pt]1{\times}1,\,256\end{smallmatrix}\right]{\times}3$};
\node[box, fill=orange!18, right=0.4cm of c2] (c3) {conv3\_x\\$[\,\cdots,512\,]{\times}4$};
\node[box, fill=orange!24, right=0.4cm of c3] (c4) {conv4\_x\\$[\,\cdots,1024\,]{\times}6$};
\node[box, fill=orange!32, right=0.4cm of c4] (c5) {conv5\_x\\$[\,\cdots,2048\,]{\times}3$};
\node[box, fill=green!12, right=0.4cm of c5] (gap) {global\\avg pool};
\node[box, fill=red!12, right=0.4cm of gap] (fc) {FC\\5 (ICDR)};
\foreach \a/\b in {in/c1,c1/mp,mp/c2,c2/c3,c3/c4,c4/c5,c5/gap,gap/fc}{\draw[arr] (\a)--(\b);}
\foreach \n/\d in {c1/{$112^2$},mp/{$56^2$},c3/{$28^2$},c4/{$14^2$},c5/{$7^2$},gap/{2048-d}}{\node[dim, below=2pt of \n] {\d};}
\coordinate (bc) at ($(c2.south)+(0,-1.9)$);
\begin{scope}[font=\tiny,
   cb/.style={draw, rounded corners=1.5pt, fill=blue!12, text width=1.7cm, align=center, minimum height=0.5cm, inner sep=1.5pt},
   ar/.style={-{Stealth[length=1.5mm]}, semithick}]
\node (x) at (bc) {$x$};
\node[cb, below=2.5mm of x] (l1) {$1{\times}1$ conv, 64};
\node[cb, below=2.5mm of l1] (l2) {$3{\times}3$ conv, 64};
\node[cb, below=2.5mm of l2] (l3) {$1{\times}1$ conv, 256};
\node[draw, circle, below=2.5mm of l3, inner sep=0.8pt] (sum) {$+$};
\node[below=2.5mm of sum] (out) {ReLU};
\foreach \a/\b in {x/l1,l1/l2,l2/l3,l3/sum,sum/out}{\draw[ar] (\a)--(\b);}
\draw[ar] (x.east) -- ++(1.0,0) |- (sum.east);
\node[right=1.05cm of l2, align=left] {identity\\shortcut $x$};
\node[below=3pt of out, font=\scriptsize] {Blok residual bottleneck: $y=\mathcal{F}(x)+x$};
\end{scope}
\draw[arr, dashed, gray] (c2.south) -- (x.north);
\end{tikzpicture}%
}
\caption{Arsitektur ResNet-50: konvolusi $7\times7$ dan \emph{max pooling} yang diikuti empat tahap residual (conv2\_x hingga conv5\_x) yang dibangun dari blok \emph{bottleneck}, lalu \emph{global average pooling} dan satu lapis \emph{fully connected}. Sisipan menunjukkan blok residual \emph{bottleneck} dengan \emph{identity shortcut}-nya, $y=\mathcal{F}(x)+x$. Dalam penelitian ini kepala klasifikasi mengeluarkan lima kelas ICDR. Sumber: diadaptasi dari He et al. (2016).}
\label{fig:resnet}
\end{figure}

#### 2.4.2 Vision Transformer (ViT)

*Vision Transformer* (ViT; Dosovitskiy et al., 2021) membawa arsitektur Transformer (Vaswani et al., 2017) ke pengenalan citra. ViT memotong citra menjadi *patch* berukuran tetap ($16 \times 16$ piksel untuk ViT-B/16), meratakan setiap *patch* menjadi sebuah *token*, menyematkan *positional embedding*, dan melewatkan *token* melalui tumpukan blok *Transformer encoder*. Di jantungnya terdapat *self-attention*, yang memungkinkan setiap *token* mengukur seberapa relevan semua *token* lainnya melalui proyeksi *query* $Q$, *key* $K$, dan *value* $V$: $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^{\top}/\sqrt{d_k}) V$. Memodelkan dependensi jarak jauh semacam itu cocok untuk *grading* DR, tempat lesi dapat tersebar di kuadran retina yang berbeda-beda. Gambar 2.3 memaparkan arsitektur ViT-B/16 beserta blok *Transformer encoder* dan *multi-head self-attention*-nya. Anupama et al. (2025) menyebut ViT secara tegas sebagai arah yang layak ditempuh untuk *grading* DR.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gambar/vit_paper.png}
\caption{Arsitektur \emph{Vision Transformer}. Sebuah citra dipecah menjadi \emph{patch} berukuran tetap yang, bersama \emph{position embedding} dan sebuah \emph{token} [class] yang dapat dipelajari, diproses oleh sebuah \emph{Transformer encoder}; panel kanan menunjukkan blok \emph{encoder} dengan \emph{multi-head self-attention}. Dalam penelitian ini masukannya adalah citra fundus dan kepala klasifikasi mengeluarkan lima kelas ICDR. Sumber: Dosovitskiy et al. (2021); blok \emph{encoder} mengikuti Vaswani et al. (2017).}
\label{fig:vit}
\end{figure}

#### 2.4.3 Transfer Learning

Dataset DR publik berukuran kecil (IDRiD hanya menawarkan beberapa ratus citra latih), sehingga memulai salah satu *backbone* dari bobot acak akan mengundang *overfitting*. Sebagai gantinya, *transfer learning* memulai dari bobot pra-latih ImageNet yang telah menangkap fitur visual umum dan melakukan *fine-tuning* pada data target setelah mengganti kepala pengklasifikasi menjadi lima kelas, yang menurunkan kebutuhan data dan mempercepat konvergensi cukup jauh untuk membuat pelatihan pada IDRiD menjadi praktis.

### 2.5 Metrik Evaluasi Klasifikasi DR Grading

#### 2.5.1 Akurasi, Presisi, Recall, dan Macro-F1

Akurasi, yaitu proporsi prediksi benar $\mathrm{Accuracy} = (1/N) \sum_{i=1}^{N} \mathbb{1}[\hat{y}_{i} = y_{i}]$, menyesatkan pada data yang tidak seimbang tempat tingkat 0 mendominasi, seperti pada IDRiD dan DDR. Karena itu penelitian ini juga melaporkan presisi dan *recall* per kelas bersama *macro-F1*, yaitu rata-rata tak berbobot atas semua kelas dari $F1 = 2\,(\mathrm{precision} \cdot \mathrm{recall}) / (\mathrm{precision} + \mathrm{recall})$. Dengan membobot setiap kelas secara setara, *macro-F1* tetap responsif terhadap seberapa baik kelas minoritas ditangani.

#### 2.5.2 Quadratic-weighted Kappa (QWK)

Karena ICDR bersifat ordinal, kesalahan dua tingkat lebih buruk daripada kesalahan satu tingkat, sebuah perbedaan yang diabaikan akurasi dan F1. *Quadratic-weighted kappa* (Cohen, 1968) menerapkan penalti yang tumbuh seiring kuadrat jarak antarkelas,

$$
\kappa_{w} \;=\; 1 - \frac{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} O_{ij}}{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} E_{ij}},
\qquad w_{ij} = \frac{(i - j)^{2}}{(K - 1)^{2}},
$$

dengan $O$ adalah matriks konfusi teramati dan $E$ adalah matriks konfusi yang diharapkan seandainya prediksi dan label saling bebas. Di sini $\kappa_w = 1$ menandakan kesepakatan sempurna dan $\kappa_w = 0$ berpadanan dengan tebakan setingkat kebetulan. Karena telah mapan sebagai skor standar dalam tantangan *grading* DR (*Kaggle Diabetic Retinopathy Detection*, APTOS 2019), QWK menjadi metrik utama penelitian ini.

#### 2.5.3 Gradient-weighted Class Activation Mapping (Grad-CAM)

*Gradient-weighted Class Activation Mapping* (Grad-CAM; Selvaraju et al., 2017) menghasilkan peta panas (*heatmap*) atas wilayah yang paling memengaruhi klasifikasi, yang diperoleh dengan membobot peta fitur lapis konvolusi terakhir menggunakan gradien skor kelas target. Untuk *grading* DR, Grad-CAM membantu memastikan apakah sebuah prediksi berlandaskan lesi yang bermakna secara klinis (seperti mikroaneurisma atau perdarahan) alih-alih artefak seperti tepi lensa. Penelitian ini memperlakukan Grad-CAM sebagai alat interpretasi pendukung yang diterapkan pada konfigurasi terbaik alih-alih sebagai perhatian kuantitatif utama, sejalan dengan rekomendasi Anupama et al. (2025).

### 2.6 Penelitian Terkait

Sebagaimana dicatat pada Bagian 1.1, *grading* DR berbasis *deep learning* mencapai kematangan klinis dan regulatori melalui sistem-sistem tonggak (Gulshan et al., 2016; Abramoff et al., 2018); pelepasan *benchmark* publik berikutnya seperti IDRiD (Porwal et al., 2020) dan DDR (Li et al., 2019) kemudian mengalihkan perhatian ke perbandingan terbuka antar-*backbone* (ResNet, DenseNet, Inception, EfficientNet) dan, yang lebih baru, ke *foundation model* seperti RETFound (Zhou et al., 2023). Penelitian ini termasuk dalam tradisi *open-benchmark* itu tetapi berpusat pada dua pertanyaan yang masih kurang tereksplorasi: bagaimana *preprocessing* dan *backbone* berinteraksi, dan apakah perbandingan yang dihasilkan berlaku antardataset.

Chokuwa dan Khan (2025) menunjukkan bahwa model *grading* DR kehilangan kinerja secara substansial ketika model yang dilatih pada satu dataset diuji pada dataset lain yang pengaturan akuisisinya berbeda, dengan penyebabnya adalah pergeseran domain (kamera, pencahayaan, demografi) alih-alih anotasi yang tidak konsisten. Hasil itulah yang membuat penelitian ini menolak memindahkan satu model antardataset dan sebaliknya memperlakukan setiap dataset sebagai *benchmark* mandiri, dengan menanyakan apakah putusan komparatifnya, yaitu *preprocessing* dan *backbone* mana yang keluar sebagai terbaik, tetap sama antardataset.

Makalah acuan penelitian ini adalah Anupama et al. (2025) di *Scientific Reports*, yang menilai beberapa *backbone* untuk *grading* DR pada satu dataset dan, pada bagian *future work*-nya, menunjuk tiga arah: (i) studi sistematis atas kombinasi *preprocessing*; (ii) validasi pada dataset *multi-center* yang mencakup perangkat dan demografi yang beragam; dan (iii) penambahan alat interpretasi seperti Grad-CAM. Penelitian ini membangun di atas ketiganya, dengan membandingkan lima teknik *preprocessing* (Bagian 2.3) pada dua *backbone* dari paradigma yang berbeda (Bagian 2.4), mengevaluasinya secara independen pada enam dataset publik (Bagian 2.2), dan menerapkan Grad-CAM pada konfigurasi terbaik (Bagian 2.5.3).

### 2.7 Kerangka Berpikir

Citra fundus mentah bervariasi dalam pencahayaan, kontras, dan warna, dan lesi DR paling awal berukuran kecil serta berkontras rendah (Bagian 2.1 dan 2.2). Karena kelima teknik *preprocessing* (Bagian 2.3) dan kedua *backbone* (Bagian 2.4) beroperasi melalui mekanisme dan *inductive bias* yang berbeda, kinerja dipandang sebagai fungsi dari interaksi *preprocessing* $\times$ *backbone*, yang justru dirancang untuk diisolasi oleh desain faktorial $5 \times 2$ (Bagian 3.1). Untuk memeriksa apakah interaksi itu melampaui satu dataset, kesepuluh konfigurasi dilatih dan dievaluasi secara independen pada enam dataset yang berbeda dalam kamera, populasi, dan pencahayaan namun berbagi ontologi ICDR, sehingga sebuah konfigurasi yang menempati peringkat pertama secara konsisten pada keenamnya dapat mengklaim validitas eksternal. Karena skala ICDR bersifat ordinal dan kelasnya tidak seimbang, perbandingannya bersandar pada QWK dan *macro-F1* alih-alih akurasi semata (Bagian 2.5) dan menggabungkan QWK per dataset melalui uji Friedman, dengan konfigurasi-konfigurasi terdepan kemudian dibandingkan lewat uji *Wilcoxon signed-rank* dan uji McNemar per dataset (Bagian 3.7). Pertimbangan-pertimbangan ini mengarah pada hipotesis yang dirumuskan pada Bagian 2.8.

### 2.8 Hipotesis Penelitian

Mengikuti rumusan masalah, penelitian ini dipandu oleh satu hipotesis statistik, yang diuji sebagaimana dijelaskan pada Bagian 3.7. Untuk sepasang konfigurasi mana pun, yang dilambangkan Konfigurasi A dan Konfigurasi B, hipotesisnya didefinisikan sebagai berikut.

- $H_{0(A,B)}$: median selisih antara skor QWK Konfigurasi A dan Konfigurasi B pada keenam dataset adalah nol; artinya, tidak ada konfigurasi yang cenderung mengungguli yang lain, dan setiap selisih yang teramati disebabkan oleh kebetulan.
- $H_{1(A,B)}$: median selisih antara skor QWK Konfigurasi A dan Konfigurasi B pada keenam dataset tidak sama dengan nol; artinya, satu konfigurasi secara sistematis berkinerja lebih baik daripada yang lain pada seluruh dataset.

*Trade-off* antara kinerja prediktif dan biaya komputasi (RM3) dikaji secara deskriptif melalui pengukuran langsung dan karena itu tidak dinyatakan sebagai hipotesis statistik.

## METODE PENELITIAN

### 3.1 Desain Penelitian

Penelitian ini mengadopsi desain eksperimen faktorial kuantitatif. Dua faktor bebasnya adalah teknik *preprocessing* citra fundus pada lima taraf (CLAHE, Ben Graham Normalization, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan MCIE) dan arsitektur *backbone* pada dua taraf (ResNet-50 dan ViT-B/16), yang berpadu menjadi sepuluh konfigurasi ($5 \times 2$); definisi operasionalnya tampil pada Bagian 3.3. Masing-masing dari keenam dataset (IDRiD, DDR, APTOS 2019, Messidor-2, EyePACS, dan DeepDRiD) berperan sebagai *benchmark* independen: pada setiap dataset, kesepuluh konfigurasi dilatih dan diuji pada partisi dataset itu sendiri di bawah satu protokol yang identik dan dinilai dengan akurasi, *quadratic-weighted kappa* (QWK), *macro-F1*, serta matriks konfusi per kelas. Luarannya adalah sebuah matriks skor QWK per dataset berukuran $10 \times 6$.

Untuk menentukan konfigurasi yang terbaik bukan pada satu dataset melainkan secara konsisten pada keenamnya, skor QWK per dataset diagregasi dengan uji Friedman (Demsar, 2006) sebagai uji *omnibus*, dan dua konfigurasi terdepan kemudian dibandingkan antardataset dengan uji *Wilcoxon signed-rank* serta di dalam setiap dataset dengan uji McNemar berkoreksi Holm-Bonferroni pada prediksi berpasangan (Bagian 3.7). Seluruh *hyperparameter*, *pipeline* augmentasi, pembagian data per dataset, dan *seed* acak dijaga identik antarkonfigurasi, sehingga perbedaan yang teramati di dalam satu dataset dapat diatribusikan semata pada teknik *preprocessing* dan pilihan *backbone*. Bobot model, *seed*, skrip, dan catatan metrik disimpan dalam repositori Git publik demi *reproducibility*. Gambar 3.1 menyajikan alur enam tahap, mulai dari pembagian per dataset melalui *preprocessing*, pelatihan, dan evaluasi hingga ke agregasi statistik dan Grad-CAM pada konfigurasi terbaik.

\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
  font=\footnotesize,
  db/.style={draw, cylinder, shape border rotate=90, aspect=0.25, fill=gray!10, align=center, font=\scriptsize, minimum width=1.5cm, minimum height=1.4cm},
  pbox/.style={draw, rounded corners=2pt, fill=orange!15, align=center, font=\scriptsize, text width=2.1cm, minimum height=0.55cm, inner sep=2pt},
  bbox/.style={draw, rounded corners=2pt, fill=cyan!18, align=center, font=\scriptsize, text width=2.0cm, minimum height=0.7cm, inner sep=2pt},
  sbox/.style={draw, rounded corners=2pt, fill=gray!8, align=center, font=\scriptsize, text width=2.4cm, minimum height=0.7cm, inner sep=2pt},
  obox/.style={draw, rounded corners=2pt, fill=green!14, align=center, font=\scriptsize, text width=2.4cm, minimum height=0.7cm, inner sep=2pt},
  arr/.style={-{Stealth[length=1.8mm]}, semithick},
  fan/.style={-{Stealth[length=1.4mm]}, thin, gray!65}
]
% ---- Baris 1: eksperimen faktorial per dataset ----
\node[db] (data) at (0,0) {Masing-masing\\6 dataset};
\node[sbox] (basic) at (2.7,0) {Pipeline dasar:\\crop, resize\\$224^2$, augment};
\node[pbox] (p1) at (5.9, 3.0) {CLAHE};
\node[pbox] (p2) at (5.9, 1.5) {Ben Graham};
\node[pbox] (p3) at (5.9, 0.0) {Adaptive Sigmoid};
\node[pbox] (p4) at (5.9,-1.5) {LAB-ACE};
\node[pbox] (p5) at (5.9,-3.0) {MCIE};
\node[bbox] (b1) at (9.4, 1.1) {ResNet-50};
\node[bbox] (b2) at (9.4,-1.1) {ViT-B/16};
\node[sbox] (conf) at (12.4,0) {$5 \times 2 = 10$\\konfigurasi};
\node[sbox] (perds) at (15.3,0) {Latih \& uji\\di tiap dataset\\$\to$ QWK};
\node[font=\scriptsize\itshape, text=gray] at (5.9,3.95) {Faktor 1: preprocessing};
\node[font=\scriptsize\itshape, text=gray] at (9.4,2.15) {Faktor 2: backbone};
\draw[arr] (data) -- (basic);
\foreach \p in {p1,p2,p3,p4,p5}{\draw[fan] (basic.east) -- (\p.west);}
\foreach \p in {p1,p2,p3,p4,p5}{\draw[fan] (\p.east) -- (b1.west); \draw[fan] (\p.east) -- (b2.west);}
\draw[arr] (b1.east) -- (conf.north west);
\draw[arr] (b2.east) -- (conf.south west);
\draw[arr] (conf) -- (perds);
% ---- Baris 2: agregasi pada keenam dataset ----
\node[sbox] (matrix) at (15.3,-5.3) {Matriks QWK\\$10 \times 6$};
\node[sbox] (fried) at (11.1,-5.3) {Uji Friedman\\(omnibus)};
\node[obox] (cd) at (7.0,-5.3) {Wilcoxon $+$ McNemar\\pada 2 konfig teratas};
\node[obox] (best) at (2.9,-5.3) {Konfigurasi terbaik\\$+$ Grad-CAM};
\node[font=\scriptsize\itshape, text=gray] at (15.3,-3.9) {ulangi untuk keenam dataset};
\draw[arr] (perds.south) -- (matrix.north);
\draw[arr] (matrix) -- (fried);
\draw[arr] (fried) -- (cd);
\draw[arr] (cd) -- (best);
\end{tikzpicture}%
}
\caption{Desain penelitian. Untuk masing-masing dari keenam dataset, sebuah citra melewati \emph{pipeline} dasar lalu bercabang ke lima teknik \emph{preprocessing} (Faktor 1); setiap cabang dipasangkan dengan kedua \emph{backbone} (Faktor 2), membentuk sepuluh konfigurasi yang dilatih dan diuji pada dataset itu untuk menghasilkan QWK-nya. Mengulang hal ini untuk keenam dataset menghasilkan matriks QWK $10 \times 6$, yang diagregasi dengan uji Friedman lalu dikaji dengan uji \emph{Wilcoxon signed-rank} dan McNemar pada dua konfigurasi terdepan untuk mengidentifikasi konfigurasi terbaik, yang akhirnya diverifikasi dengan Grad-CAM.}
\label{fig:pipeline}
\end{figure}

### 3.2 Dataset

#### 3.2.1 Partisi Latih, Validasi, dan Uji per Dataset

Masing-masing dari keenam dataset dibagi menjadi partisi latih, validasi, dan uji secara independen, dan kesepuluh konfigurasi dilatih serta dievaluasi di dalam setiap dataset menggunakan partisi-partisi tersebut. Partisi validasi digunakan untuk *early stopping* dan pemilihan *checkpoint*; partisi uji diakses hanya sekali per konfigurasi, di akhir pelatihan, untuk menghasilkan metrik yang dilaporkan. Untuk dataset yang menyediakan partisi resmi, partisi itu diadopsi tanpa perubahan; untuk yang lain, digunakan pembagian terstratifikasi di bawah *seed* tetap.

Untuk **IDRiD**, partisi resmi (413 citra latih, 103 uji; Porwal et al., 2020) digunakan, dengan 15% citra latih (62 citra) disisihkan sebagai partisi validasi terstratifikasi di bawah *seed* tetap, menyisakan 351 citra latih. Untuk **DDR** (Li et al., 2019), partisi latih, validasi, dan uji resmi digunakan setelah citra *ungradable* disaring keluar. Untuk **DeepDRiD** (Liu et al., 2022) dan **EyePACS** (Kaggle dan EyePACS, 2015), partisi resmi juga diadopsi; untuk EyePACS, citra *ungradable* dihilangkan dan sebuah subset terstratifikasi per kelas diambil agar biaya pelatihan tetap terkelola sembari mempertahankan distribusi kelas. Untuk **APTOS 2019** (APTOS, 2019) dan **Messidor-2** (Decenciere et al., 2014), yang tidak menyediakan partisi latih dan uji resmi, digunakan pembagian terstratifikasi 70/15/15 latih/validasi/uji di bawah *seed* tetap. Semua pembagian dirilis dalam repositori publik agar dapat direproduksi secara persis.

#### 3.2.2 Distribusi Kelas dan Penanganan Ketidakseimbangan

Keenam dataset tidak seimbang (*long-tailed*): tingkat 0 mendominasi sementara tingkat yang lebih tinggi, terutama tingkat 3 dan 4, merupakan minoritas. Di dalam setiap dataset, ketidakseimbangan ditangani secara identik pada dua tingkat, yaitu (i) *weighted random sampler* dengan bobot $\propto 1/\sqrt{n_k}$ ($n_k$ adalah jumlah citra latih kelas $k$ pada dataset tersebut) di setiap *mini-batch*, dan (ii) *categorical cross-entropy* berbobot kelas dengan bobot $\propto 1/\sqrt{n_k}$ yang dinormalkan agar berjumlah $K$. QWK dipilih sebagai metrik pemilihan *checkpoint* karena peka terhadap jarak ordinal dan relatif tahan terhadap distribusi marginal.

### 3.3 Variabel dan Definisi Operasional Variabel

Penelitian ini melibatkan dua variabel bebas, tiga variabel terikat, dan sekumpulan variabel kontrol, yang didefinisikan secara operasional pada Tabel 3.1.

\begin{table}[htbp]
\caption{Variabel penelitian dan definisi operasionalnya.}
\label{tab:variabel}
\begin{center}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|p{3.1cm}|p{2.5cm}|p{7.2cm}|}
\hline
\textbf{Variabel} & \textbf{Jenis} & \textbf{Definisi operasional} \\
\hline
Teknik \emph{preprocessing} (bebas) & Kategorikal, 5 taraf & Transformasi intensitas atau warna yang diterapkan pada setiap citra setelah \emph{cropping} dan sebelum \emph{resizing}; taraf dan parameter pada Bagian 3.4. \\
\hline
Arsitektur \emph{backbone} (bebas) & Kategorikal, 2 taraf & Jaringan ekstraksi fitur (ResNet-50 atau ViT-B/16) yang diinisialisasi dari bobot ImageNet dan di-\emph{fine-tune} pada data \emph{grading} DR; Bagian 3.5. \\
\hline
Kinerja klasifikasi (terikat) & Kontinu & Akurasi, QWK, dan \emph{macro-F1} pada partisi uji (Bagian 2.5). \\
\hline
Konsistensi lintas-dataset (terikat) & Ordinal/Kontinu & Peringkat rata-rata setiap konfigurasi pada keenam dataset dan signifikansinya di bawah uji Friedman, uji \emph{Wilcoxon signed-rank}, serta uji McNemar per dataset, sebagai ukuran seberapa konsisten sebuah konfigurasi berkinerja antardataset (Bagian 3.7). \\
\hline
Biaya komputasi (terikat) & Kontinu & Jumlah parameter model dan rata-rata waktu inferensi per citra. \\
\hline
Kontrol & Tetap & Pipeline dasar (Bagian 3.4.1), augmentasi (Bagian 3.4.3), \emph{optimizer} dan \emph{hyperparameter} (Bagian 3.6), pembagian data, \emph{seed} acak, dan lingkungan komputasi (Bagian 3.8), dijaga identik antarkonfigurasi. \\
\hline
\end{tabular}
\end{center}\end{table}

### 3.4 Preprocessing Data

#### 3.4.1 Pipeline Dasar

Sebelum langkah spesifik teknik apa pun, setiap citra pada keenam dataset melewati satu *pipeline* dasar bersama: (i) *mask* retina diestimasi dengan *adaptive thresholding* pada *green channel* lalu dipangkas ke kotak pembatas minimumnya untuk menghilangkan bingkai hitam; (ii) citra di-*resize* menjadi $224 \times 224$ piksel dengan interpolasi bilinear agar sesuai dengan masukan yang diharapkan bobot ImageNet; dan (iii) nilai diskalakan ke $[0, 1]$ lalu distandarkan per kanal dengan rata-rata dan simpangan baku ImageNet ($\mu = (0.485, 0.456, 0.406)$, $\sigma = (0.229, 0.224, 0.225)$). *Pipeline* yang sama mengatur pelatihan, validasi, dan pengujian pada setiap dataset, sehingga tidak ada perbedaan konfigurasi yang tersembunyi di balik perbedaan *preprocessing* dasar.

#### 3.4.2 Penerapan Lima Teknik Preprocessing

Teknik *preprocessing* diterapkan di antara *cropping* dan *resizing*; parameter setiap teknik (konsepnya pada Bagian 2.3) adalah sebagai berikut. **CLAHE**: *clip limit* 2,0 dan *tile* $8 \times 8$ pada kanal L (LAB). **Ben Graham Normalization**: $\alpha = 4$, jari-jari Gaussian $\sigma$ sebesar 10% diameter retina, dan $\gamma = 128$, mengikuti resep asli Kaggle 2015. **Adaptive Sigmoid Enhancement**: $\alpha$ dan $\beta$ dihitung secara adaptif dari rata-rata dan simpangan baku intensitas *patch* lokal. **LAB-ACE**: CLAHE pada kanal L yang diikuti rekonstruksi ke RGB. **MCIE**: kombinasi *green channel* asli, hasil CLAHE pada kanal L, dan hasil Ben Graham Normalization menjadi satu citra tiga kanal.

#### 3.4.3 Augmentasi Data

Augmentasi *online* diterapkan hanya pada partisi latih setiap dataset, setelah *preprocessing*. Augmentasi geometris terdiri atas *flip* horizontal dan vertikal (peluang 0,5) dan rotasi acak $[-30^{\circ}, +30^{\circ}]$, yang mempertahankan label karena tingkat ICDR tidak bergantung pada orientasi spasial. Augmentasi fotometrik terdiri atas *jitter* kecerahan dan kontras acak $[-0.2, +0.2]$ untuk mensimulasikan variasi pencahayaan antarlokasi. Partisi validasi dan uji setiap dataset tidak diaugmentasi.

### 3.5 Arsitektur Model

Konsep kedua *backbone* dan diagram arsitekturnya telah dijelaskan pada Bagian 2.4 (Gambar 2.2 dan 2.3); bagian ini merinci hanya konfigurasi eksperimennya. ResNet-50 dimuat dengan bobot pra-latih ImageNet-1k melalui torchvision 0.18, sedangkan ViT-B/16 dimuat dengan bobot pra-latih ImageNet-21k melalui timm 1.0. Pada kedua *backbone*, kepala klasifikasi asli diganti dengan satu lapis linear baru menuju lima *logit* ICDR (ResNet-50: $2048 \to 5$ dari vektor *global average pooling*; ViT-B/16: $768 \to 5$ dari *token* CLS). Semua parameter dilatih bersama (*full fine-tuning*) tanpa membekukan lapis mana pun, dengan keluaran *softmax* yang dioptimalkan menggunakan *categorical cross-entropy* berbobot kelas (Bagian 3.2.2 dan 3.6).

### 3.6 Protokol Pelatihan

Setiap model dioptimalkan dengan AdamW (Loshchilov dan Hutter, 2019) pada *learning rate* awal $\eta_{0} = 1 \times 10^{-4}$, *weight decay* $1 \times 10^{-4}$, $\beta_{1} = 0.9$, dan $\beta_{2} = 0.999$. Setelah *warm-up* linear selama tiga *epoch*, *learning rate* meluruh pada jadwal kosinus hingga $\eta_{\min} = 1 \times 10^{-6}$. Setiap *batch* memuat 16 citra, yang muat dalam memori satu GPU kelas menengah (sekitar 16 GB). Setiap proses berlangsung paling lama 50 *epoch* dan berhenti dini berdasarkan QWK validasi (*patience* 10 *epoch*), dan *checkpoint* dengan QWK validasi terbaik adalah yang dibawa ke evaluasi uji.

Fungsi rugi (*loss*) adalah *categorical cross-entropy* berbobot kelas (bobot seperti pada Bagian 3.2.2). Mixup, *label smoothing*, dan *focal loss* sengaja tidak digunakan, sehingga perbedaan kinerja antarkonfigurasi murni berasal dari teknik *preprocessing* dan pilihan *backbone*. Semua *random number generator* diinisialisasi dengan *seed* yang sama agar hasilnya dapat direproduksi. Karena desainnya melatih setiap konfigurasi pada setiap dataset, terdapat sepuluh konfigurasi pada enam dataset, sehingga totalnya enam puluh proses pelatihan. Agar tetap terkelola, dataset EyePACS yang sangat besar dilatih pada subset terstratifikasi per kelas (Bagian 3.2.1), dan dataset yang lebih kecil menyumbang sebagian besar proses dengan biaya rendah; anggaran pelatihan total secara indikatif karena itu berada pada orde ratusan GPU-jam pada satu GPU, yang dijadwalkan sepanjang bulan-bulan pelatihan dalam rencana kerja (Bagian 3.9).

### 3.7 Protokol Evaluasi

Pada masing-masing dari keenam dataset, partisi uji dievaluasi satu kali per konfigurasi dengan akurasi, QWK, *macro-F1*, dan matriks konfusi per kelas, semuanya dilaporkan dalam tabel hasil per dataset. Hal ini menghasilkan sebuah matriks skor QWK berukuran $10 \times 6$, satu per konfigurasi per dataset, yang menjadi dasar analisis statistik.

Analisis statistik berjalan dalam tiga tahap: uji *omnibus*, pemeringkatan deskriptif, dan uji berpasangan konfirmatori pada konfigurasi-konfigurasi terdepan. Sebagai tahap *omnibus*, kesepuluh konfigurasi dibandingkan antardataset dengan uji Friedman (Demsar, 2006), prosedur non-parametrik standar untuk membandingkan beberapa metode pada banyak dataset. Di dalam setiap dataset, kesepuluh konfigurasi diperingkat berdasarkan QWK, dan uji Friedman menguji apakah peringkat rata-ratanya berbeda secara signifikan secara keseluruhan,

$$\chi_F^2 = \frac{12N}{k(k+1)}\left[\sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4}\right],$$

dengan $N=6$ dataset, $k=10$ konfigurasi, dan $R_j$ adalah peringkat rata-rata konfigurasi $j$. QWK adalah metrik yang menjadi dasar uji karena merupakan metrik utama yang peka-ordinal untuk *grading* DR; akurasi dan *macro-F1* dilaporkan secara deskriptif di sampingnya. Ketika uji Friedman signifikan, urutan peringkat rata-rata kesepuluh konfigurasi dilaporkan secara deskriptif untuk menunjukkan konfigurasi mana yang berkinerja terbaik antardataset. Karena enam dataset memberi uji *omnibus* daya yang terbatas untuk memilah kesepuluh konfigurasi, pemeringkatan ini diperlakukan sebagai bukti deskriptif atas konsistensi alih-alih sebagai sekumpulan klaim signifikansi berpasangan, dan pengujian konfirmatori di bawah ini dibatasi pada dua konfigurasi terdepan.

Dua konfigurasi terdepan (yang memiliki peringkat rata-rata terbaik antardataset) kemudian dibandingkan secara langsung pada dua tingkat. Antardataset, keduanya dibandingkan dengan uji *Wilcoxon signed-rank* (Wilcoxon, 1945) pada enam nilai QWK berpasangan per datasetnya,

$$W = \min(W^+, W^-), \qquad W^+ = \sum_{d_i > 0} \operatorname{rank}(|d_i|),$$

dengan $d_i$ adalah selisih QWK antara kedua konfigurasi pada dataset $i$ dan peringkat diambil atas $|d_i|$; hipotesis nolnya adalah median selisih sama dengan nol. Ini adalah satu perbandingan terencana atas dua metode, yang untuknya enam dataset berpasangan memberikan daya yang memadai. Di dalam setiap dataset, kedua konfigurasi yang sama dibandingkan dengan uji McNemar (McNemar, 1947) pada prediksi per citra berpasangan dari partisi uji dataset tersebut,

$$\chi^2 = \frac{(b - c)^2}{b + c},$$

dengan $b$ dan $c$ menghitung citra yang diklasifikasikan benar oleh satu konfigurasi dan salah oleh yang lain; karena beroperasi pada ribuan sampel berpasangan, uji ini memiliki daya yang jauh lebih tinggi di dalam satu dataset. Karena diterapkan sekali per dataset, keluarga enam uji McNemar per dataset dikoreksi dengan prosedur Holm-Bonferroni (Holm, 1979) untuk mengendalikan *family-wise error rate*.

Selain membandingkan konfigurasi secara utuh, desain faktorial juga memungkinkan setiap faktor diuji secara terpisah, yang memetakan langsung ke RM1 dan menjaga jumlah perbandingan tetap cukup kecil agar enam dataset mempertahankan daya yang berguna. Untuk faktor *backbone*, skor QWK per dataset dirata-ratakan atas lima teknik *preprocessing* untuk memberi satu nilai per *backbone* pada setiap dataset, lalu ResNet-50 dan ViT-B/16 dibandingkan antar keenam dataset dengan uji *Wilcoxon signed-rank*, sebuah perbandingan terencana tunggal. Untuk faktor *preprocessing*, skor QWK per dataset dirata-ratakan atas dua *backbone* untuk memberi matriks $5 \times 6$, yang di atasnya kelima teknik *preprocessing* dibandingkan dengan uji Friedman; ketika signifikan, setiap teknik dibandingkan terhadap *baseline* dengan uji *Wilcoxon signed-rank* berpasangan di bawah koreksi Holm-Bonferroni. Interaksi *preprocessing* $\times$ *backbone*, yaitu apakah teknik *preprocessing* terbaik berbeda di antara kedua *backbone*, dikaji secara deskriptif dengan membandingkan pemeringkatan *preprocessing* di dalam setiap *backbone* secara terpisah.

Pembagian tugasnya eksplisit: uji Wilcoxon antardataset menjawab apakah konfigurasi terdepan konsisten lebih baik pada keenam dataset, sedangkan uji McNemar dalam-dataset menjawab apakah kedua konfigurasi berbeda pada suatu dataset tertentu. Sebuah konfigurasi dianggap unggul secara kokoh hanya ketika ia sekaligus memimpin pemeringkatan lintas-dataset dan memenangkan perbandingan McNemar dalam-dataset. Mengikuti prinsip menetapkan luaran di muka, jika tidak ada konfigurasi yang memisahkan diri dari kelompoknya, kesimpulan bahwa konfigurasi terbaik bersifat spesifik-dataset dilaporkan sebagai hasil substantif yang konsisten dengan motivasi penelitian (Bagian 1.1), bukan sebagai temuan yang tidak konklusif.

Analisis diimplementasikan dengan pustaka \texttt{scipy.stats} dan \texttt{statsmodels} (uji Friedman dan *Wilcoxon signed-rank*, uji McNemar, serta koreksi Holm-Bonferroni). Selain itu, Grad-CAM (Selvaraju et al., 2017) dibangkitkan pada konfigurasi terbaik untuk verifikasi visual atas dasar prediksinya.

### 3.8 Alat dan Lingkungan Implementasi

*Pipeline* eksperimen diimplementasikan dalam Python 3.11 menggunakan pustaka berikut: PyTorch 2.3 sebagai *backend* tensor dan *autodiff*; torchvision 0.18 untuk ResNet-50 dan bobot pra-latih ImageNet-1k; timm 1.0 (Wightman, 2019) untuk ViT-B/16 dan bobot pra-latih ImageNet-21k; OpenCV 4.9 dan scikit-image 0.22 untuk mengimplementasikan lima teknik *preprocessing*; Albumentations 1.4 untuk *pipeline* augmentasi; scikit-learn 1.4 untuk menghitung metrik evaluasi; SciPy 1.13 dan statsmodels 0.14 untuk uji statistik (Friedman, *Wilcoxon signed-rank*, McNemar, dan koreksi Holm-Bonferroni); serta Matplotlib 3.8 dan seaborn 0.13 untuk memvisualisasikan matriks konfusi dan *heatmap* Grad-CAM. Eksperimen dijalankan pada lingkungan satu GPU; sebuah GPU kelas menengah dengan memori sekitar 16 GB memadai untuk ukuran *batch* dan resolusi masukan yang dipilih, dan perangkat keras persisnya dapat bervariasi menurut sumber daya komputasi yang tersedia pada saat pelatihan. Seluruh kode, skrip pelatihan, skrip evaluasi, berkas konfigurasi, dan catatan metrik disimpan dalam repositori Git publik yang disertai README reproduksi dengan versi pustaka, *seed*, dan perintah eksekusi yang eksplisit.

### 3.9 Jadwal Penelitian

Kegiatan penelitian direncanakan berlangsung selama dua belas bulan, dari Maret 2026 hingga Februari 2027, mencakup studi literatur dan penulisan proposal, seminar proposal dan revisinya, penyiapan lingkungan dan pengumpulan keenam dataset, implementasi *pipeline* *preprocessing*, pelatihan kesepuluh konfigurasi pada masing-masing dari keenam dataset, evaluasi per dataset, analisis statistik Friedman dan visualisasi Grad-CAM, penulisan Bab IV dan Bab V, seminar hasil, serta sidang skripsi dan penyerahan akhir. Jadwal bulanannya dirinci pada Tabel 3.2; kolom dari Maret hingga Desember adalah bulan-bulan pada 2026, sedangkan Januari dan Februari adalah bulan-bulan pada 2027. Jadwal ini bersifat indikatif dan dapat disesuaikan menurut hasil konsultasi dengan para pembimbing serta ketersediaan sumber daya komputasi.

\begin{table}[htbp]
\caption{Jadwal penelitian untuk periode Maret 2026 hingga Februari 2027.}
\label{tab:jadwal}
\begin{center}
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{|c|p{3.3cm}|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{No} & \textbf{Kegiatan} & \textbf{Mar} & \textbf{Apr} & \textbf{Mei} & \textbf{Jun} & \textbf{Jul} & \textbf{Agu} & \textbf{Sep} & \textbf{Okt} & \textbf{Nov} & \textbf{Des} & \textbf{Jan} & \textbf{Feb} \\
\hline
1 & Studi literatur dan penulisan proposal & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  &  &  \\
\hline
2 & Seminar proposal dan revisi &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  &  \\
\hline
3 & Penyiapan \emph{dataset} dan lingkungan &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  \\
\hline
4 & Implementasi \emph{preprocessing} (5 teknik) &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  \\
\hline
5 & Pelatihan 10 konfig pada 6 dataset &  &  &  &  & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  \\
\hline
6 & Evaluasi per \emph{dataset} &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  \\
\hline
7 & Analisis Friedman dan Grad-CAM &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  \\
\hline
8 & Penulisan Bab IV &  &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  \\
\hline
9 & Penulisan Bab V dan revisi &  &  &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  \\
\hline
10 & Bimbingan (Pembimbing I dan II) & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} &  \\
\hline
11 & Seminar hasil &  &  &  &  &  &  &  &  &  &  & \cellcolor{black} &  \\
\hline
12 & Sidang skripsi dan penyerahan akhir &  &  &  &  &  &  &  &  &  &  &  & \cellcolor{black} \\
\hline
\end{tabular}
\end{center}\end{table}



## DAFTAR PUSTAKA {.unnumbered}

\refitem{Abramoff, M. D., Lavin, P. T., Birch, M., Shah, N., \& Folk, J. C. (2018). Pivotal Trial of an Autonomous AI-Based Diagnostic System for Detection of Diabetic Retinopathy in Primary Care Offices. \emph{NPJ Digital Medicine}, \emph{1}(1), 39.}

\refitem{American Diabetes Association. (2024). Standards of Care in Diabetes: 2024. \emph{Diabetes Care}, \emph{47}(Suppl. 1), S1--S322.}

\refitem{Anupama, B. C., Rao, S. N., Malini, M. B., \& Athreya, V. V. (2025). Comparative Analysis of Novel Preprocessing Techniques and Deep Learning Based Multi-Modal Feature Fusion for Diabetic Retinopathy Grading. \emph{Scientific Reports}, \emph{15}, 31339.}

\refitem{Asia Pacific Tele-Ophthalmology Society (APTOS). (2019). \emph{APTOS 2019 Blindness Detection} [Data set]. Kaggle. \texttt{https://www.kaggle.com/c/aptos2019-blindness-detection}}

\refitem{Chokuwa, S., \& Khan, M. H. (2025). Divergent Domains, Convergent Grading: Enhancing Generalization in Diabetic Retinopathy Grading. \emph{IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}.}

\refitem{Chopra, M., Sparrenberg, L., Berger, A., Khanna, S., Terheyden, J. H., \& Sifa, R. (2025). From Retinal Pixels to Patients: Evolution of Deep Learning Research in Diabetic Retinopathy Screening. \emph{2025 IEEE International Conference on Big Data (IEEE BigData)}. arXiv:2511.11065}

\refitem{Cohen, J. (1968). Weighted Kappa: Nominal Scale Agreement Provision for Scaled Disagreement or Partial Credit. \emph{Psychological Bulletin}, \emph{70}(4), 213--220.}

\refitem{Decenciere, E., Zhang, X., Cazuguel, G., Lay, B., Cochener, B., Trone, C., et al. (2014). Feedback on a Publicly Distributed Image Database: The Messidor Database. \emph{Image Analysis \& Stereology}, \emph{33}(3), 231--234.}

\refitem{Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. \emph{Journal of Machine Learning Research}, \emph{7}, 1--30.}

\refitem{Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., \& Houlsby, N. (2021). An Image Is Worth 16$\times$16 Words: Transformers for Image Recognition at Scale. \emph{International Conference on Learning Representations (ICLR)}.}

\refitem{Graham, B. (2015). \emph{Kaggle Diabetic Retinopathy Detection Competition Report}. University of Warwick.}

\refitem{Gulshan, V., Peng, L., Coram, M., Stumpe, M. C., Wu, D., Narayanaswamy, A., \ldots\ \& Webster, D. R. (2016). Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. \emph{JAMA}, \emph{316}(22), 2402--2410.}

\refitem{He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep Residual Learning for Image Recognition. \emph{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 770--778.}

\refitem{Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. \emph{Scandinavian Journal of Statistics}, \emph{6}(2), 65--70.}

\refitem{Kaggle, \& EyePACS. (2015). \emph{Diabetic Retinopathy Detection} [Data set]. Kaggle. \texttt{https://www.kaggle.com/c/diabetic-retinopathy-detection}}

\refitem{Li, T., Gao, Y., Wang, K., Guo, S., Liu, H., \& Kang, H. (2019). Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening. \emph{Information Sciences}, \emph{501}, 511--522.}

\refitem{Liu, R., Wang, X., Wu, Q., Dai, L., Fang, X., Yan, T., et al. (2022). DeepDRiD: Diabetic Retinopathy-Grading and Image Quality Estimation Challenge. \emph{Patterns}, \emph{3}(6), 100512.}

\refitem{Loshchilov, I., \& Hutter, F. (2019). Decoupled Weight Decay Regularization. \emph{International Conference on Learning Representations (ICLR)}.}

\refitem{McNemar, Q. (1947). Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. \emph{Psychometrika}, \emph{12}(2), 153--157.}

\refitem{Porwal, P., Pachade, S., Kokare, M., Deshmukh, G., Son, J., Bae, W., \ldots\ \& Meriaudeau, F. (2020). IDRiD: Diabetic Retinopathy Segmentation and Grading Challenge. \emph{Medical Image Analysis}, \emph{59}, 101561.}

\refitem{Saputra, N. A., Helvinda, W., \& Rahman, K. (2024). Prevalence and Risk Factors of Diabetic Retinopathy in a Tertiary Hospital in Padang, Indonesia. \emph{Bioscientia Medicina: Journal of Biomedicine and Translational Research}, \emph{9}(1), 219--231.}

\refitem{Sasongko, M. B., Widyaputri, F., Agni, A. N., Wardhana, F. S., Kotha, S., Gupta, P., Widayanti, T. W., Haryanto, S., Widyaningrum, R., Wong, T. Y., Kawasaki, R., \& Wang, J. J. (2025). Incidence and Progression of Diabetic Retinopathy and Blindness in Indonesian Adults with Type 2 Diabetes. \emph{PLoS ONE}, \emph{20}, e0322093.}

\refitem{Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., \& Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. \emph{Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 618--626.}

\refitem{Teo, Z. L., Tham, Y. C., Yu, M., Chee, M. L., Rim, T. H., Cheung, N., \ldots\ \& Cheng, C. Y. (2021). Global Prevalence of Diabetic Retinopathy and Projection of Burden through 2045: Systematic Review and Meta-Analysis. \emph{Ophthalmology}, \emph{128}(11), 1580--1591.}

\refitem{Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., \& Polosukhin, I. (2017). Attention Is All You Need. \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 5998--6008.}

\refitem{Wightman, R. (2019). \emph{PyTorch Image Models (timm)} [Computer software]. GitHub. \texttt{https://github.com/huggingface/pytorch-image-models}}

\refitem{Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. \emph{Biometrics Bulletin}, \emph{1}(6), 80--83.}

\refitem{Wilkinson, C. P., Ferris III, F. L., Klein, R. E., Lee, P. P., Agardh, C. D., Davis, M., \ldots\ \& Verdaguer, J. T. (2003). Proposed International Clinical Diabetic Retinopathy and Diabetic Macular Edema Disease Severity Scales. \emph{Ophthalmology}, \emph{110}(9), 1677--1682.}

\refitem{Wong, T. Y., \& Sabanayagam, C. (2023). The War on Diabetic Retinopathy: Where Are We Now? \emph{Asia-Pacific Journal of Ophthalmology}, \emph{12}(3), 213--221.}

\refitem{Zhou, Y., Chia, M. A., Wagner, S. K., Ayhan, M. S., Williamson, D. J., Struyven, R. R., \ldots\ \& Keane, P. A. (2023). A Foundation Model for Generalizable Disease Detection from Retinal Images. \emph{Nature}, \emph{622}(7981), 156--163.}
