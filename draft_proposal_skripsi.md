# PROPOSAL SKRIPSI

**Judul:**
Analisis Komparatif Teknik *Preprocessing* Citra Fundus dan Arsitektur *Deep Learning* untuk Klasifikasi Tingkat Keparahan *Diabetic Retinopathy* dengan Evaluasi Lintas *Dataset*

---

## PENDAHULUAN

### 1.1 Latar Belakang

Diabetes melitus telah menjadi salah satu beban penyakit tidak menular paling mendesak pada abad ke-21, dan *diabetic retinopathy* (DR), sebuah komplikasi mikrovaskuler progresif dari penyakit tersebut, tetap menjadi penyebab utama kebutaan yang dapat dicegah pada kelompok dewasa usia produktif di mayoritas negara berpenghasilan menengah dan tinggi (American Diabetes Association, 2024). Teo et al. (2021) memproyeksikan jumlah orang dewasa yang terkena DR akan meningkat dari 103 juta pada tahun 2020 menjadi sekitar 161 juta pada tahun 2045, dengan DR yang mengancam penglihatan menjangkiti hampir 29 juta orang secara global. Situasi di Indonesia sangat mendesak: Sasongko et al. (2025), dalam studi kohort lima tahun terhadap 695 pasien diabetes tipe 2 di komunitas Yogyakarta, mencatat insidens DR sebesar 34,6 per 1.000 orang-tahun dan kebutaan terkait DR sebesar 8,3 per 1.000 orang-tahun, sementara Saputra et al. (2024) melaporkan prevalensi DR sebesar 55% di antara pasien diabetes di sebuah rumah sakit rujukan di Padang. Akan tetapi, rasio oftalmolog di Indonesia kurang dari dua spesialis per 100.000 penduduk dan terkonsentrasi di wilayah urban Pulau Jawa, sehingga secara struktural tidak memungkinkan pelaksanaan pemeriksaan fundus tahunan sebagaimana direkomendasikan oleh *Standards of Care in Diabetes* (American Diabetes Association, 2024). Strategi skrining yang layak secara operasional untuk Indonesia harus bersandar pada interpretasi otomatis citra fundus di layanan kesehatan tingkat primer.

Tugas yang perlu diotomatisasi bukan sekadar mendeteksi ada atau tidaknya penyakit, melainkan menentukan tingkat keparahannya. Secara klinis, DR distratifikasi menggunakan *International Clinical Diabetic Retinopathy* (ICDR) *severity scale* yang diajukan Wilkinson et al. (2003) ke dalam lima tingkat ordinal, mulai dari tidak ada retinopati (*no DR*), NPDR ringan, NPDR sedang, NPDR berat, hingga *proliferative DR* (PDR), dan tingkat inilah yang menentukan keputusan rujukan: pasien dengan NPDR sedang atau lebih berat dirujuk untuk evaluasi spesialis, sementara PDR menuntut intervensi segera untuk mencegah kehilangan penglihatan permanen. Karena setiap tingkat memetakan ke tindakan klinis yang berbeda, kesalahan klasifikasi tidak bersifat netral: menggrade pasien terlalu rendah berarti rujukan yang terlambat, sedangkan menggrade terlalu tinggi membebani kapasitas spesialis yang sudah langka. Sistem skrining otomatis karena itu dituntut akurat tepat pada batas-batas keputusan kelima kelas ini, bukan sekadar tinggi pada rata-rata akurasinya.

*Deep learning* telah mengubah lanskap otomatisasi DR *grading* dalam satu dekade terakhir. Gulshan et al. (2016), dalam studi *landmark* yang dipublikasikan di *JAMA*, menunjukkan bahwa *convolutional neural network* (CNN) yang dilatih pada 128.175 citra fundus mampu mendeteksi *referable DR* dengan sensitivitas dan spesifisitas di atas 90%, setara dengan oftalmolog tersertifikasi pada data terkurasi. Abramoff et al. (2018) menerjemahkan pendekatan serupa menjadi IDx-DR, perangkat kecerdasan buatan pertama yang memperoleh izin *U.S. Food and Drug Administration* untuk penggunaan klinis mandiri, sementara *foundation model* berskala besar seperti RETFound (Zhou et al., 2023) semakin mempersempit jarak antara sistem penelitian dan sistem klinis. Survei komprehensif terkini yang mensintesis lebih dari lima puluh studi dan dua puluh dataset DR sepanjang 2016 hingga 2025 menegaskan kematangan teknis ini, tetapi sekaligus menggarisbawahi bahwa celah utama telah bergeser ke validasi *multi-center* dan kepercayaan klinis (Chopra et al., 2025). Dari sisi kemampuan teknis, dengan demikian, skrining DR otomatis bukan lagi persoalan yang terbuka. Persoalannya adalah hampir seluruh capaian tersebut diperoleh dan divalidasi pada data yang terkurasi rapi, sehingga belum tentu bertahan ketika sistem dipindahkan ke kondisi lapangan yang jauh lebih beragam.

Justru di titik inilah jarak antara capaian riset dan *deployment* klinis rutin terbuka, pada hal-hal yang kerap terabaikan dalam evaluasi akademik. Isu pertama adalah kualitas citra fundus. Pencahayaan yang tidak merata, kontras rendah, dan *noise* akuisisi secara sistematis menurunkan akurasi sistem *grading* otomatis (Anupama et al., 2025), dan persoalan ini paling parah pada perangkat fundus berbiaya rendah yang realistis digunakan di layanan kesehatan primer Indonesia. Ketika lesi awal seperti mikroaneurisma tidak terbaca akibat citra yang buruk, model cenderung menggrade pasien lebih rendah dari kondisi sebenarnya, tepat jenis kesalahan yang berujung pada rujukan yang terlambat. Berbagai teknik *preprocessing* telah diusulkan untuk memperbaiki visibilitas lesi, mulai dari *Contrast Limited Adaptive Histogram Equalization* (CLAHE), Ben Graham *preprocessing*, dan ekstraksi *green channel*, hingga teknik *enhancement* yang lebih baru seperti *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*. Akan tetapi, penerapannya dalam literatur cenderung bersifat *ad-hoc*: setiap penelitian memilih satu teknik tanpa membandingkannya secara setara dengan alternatif lain pada arsitektur dan protokol pelatihan yang sama. Anupama et al. (2025), dalam publikasi mereka di *Scientific Reports*, secara eksplisit menilai bahwa metode *preprocessing* untuk DR *grading* saat ini "masih terbatas pada fokus tunggal, seperti penerapan filter *noise* atau *contrast enhancement*, yang tidak memberikan solusi menyeluruh untuk menangani keragaman dan kompleksitas citra fundus" dan merekomendasikan eksplorasi sistematis terhadap kombinasi tekniknya. Selama komparasi *apple-to-apple* antarteknik *preprocessing* pada arsitektur yang sama belum tersedia, pemilihan *preprocessing* untuk sistem skrining masih bertumpu pada dugaan alih-alih bukti.

Isu kedua adalah generalisasi lintas-dataset. Chokuwa dan Khan (2025) menunjukkan bahwa model DR *grading* yang dilatih pada satu sumber dataset performanya merosot secara signifikan ketika dihadapkan pada data di luar distribusi dari *setting* akuisisi yang berbeda, bahkan ketika ontologi *grading* yang digunakan identik. Penurunan semacam ini bukan sekadar selisih statistik: model yang tampak layak pada data pengembangan dapat berubah menjadi tidak dapat diandalkan untuk keputusan rujukan ketika diterapkan pada populasi baru. Perbedaan perangkat kamera, populasi pasien, dan protokol pencahayaan antarinstitusi menjadi penyebab utamanya, dan kondisi inilah yang persis akan dihadapi di Indonesia, ketika model dioperasikan pada perangkat kamera fundus dan populasi pasien yang berbeda dari dataset publik tempatnya dilatih. Model yang salah mengklasifikasi pasien Indonesia tidak hanya berisiko melewatkan kasus yang seharusnya dirujuk, tetapi juga memboroskan kapasitas spesialis yang langka untuk rujukan yang keliru. Anupama et al. (2025) turut menggarisbawahi keterbatasan yang sama pada bagian *future work* mereka, dan merekomendasikan bahwa "penelitian selanjutnya perlu melibatkan dataset *multi-center* dengan variasi perangkat akuisisi, latar belakang etnis, dan distribusi demografis untuk meningkatkan ketahanan dan aplikabilitas dunia nyata". Tanpa bukti empiris mengenai seberapa besar penurunan performa lintas-dataset, klaim kelayakan klinis sebuah sistem skrining menjadi sulit dipertanggungjawabkan.

Penelitian ini menjawab kedua celah tersebut dengan landasan pada rekomendasi *future work* Anupama et al. (2025). Penelitian ini membandingkan lima teknik *preprocessing* (CLAHE, Ben Graham *preprocessing*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*) yang dipasangkan dengan dua arsitektur *backbone* (ResNet-50 sebagai *baseline* konvolusional dan *Vision Transformer* sebagaimana disarankan Anupama et al., 2025), sehingga menghasilkan sepuluh konfigurasi. Seluruh konfigurasi dilatih pada IDRiD (Porwal et al., 2020) sebagai data *in-distribution*, lalu diuji pada dua kondisi, yaitu IDRiD *test set* dan DDR (Li et al., 2019) sebagai evaluasi lintas-dataset tanpa adaptasi, dengan metrik akurasi, *quadratic-weighted kappa*, dan *macro-F1*. Grad-CAM diterapkan pada konfigurasi terbaik untuk memverifikasi dasar prediksi model. Kontribusi penelitian ini ada tiga: (i) perbandingan empiris yang *reproducible* antara teknik *preprocessing* dan *backbone* pada DR *grading*, yang belum tersedia dalam literatur; (ii) kuantifikasi penurunan kinerja lintas-dataset sebagai bukti yang relevan untuk *deployment* pada populasi klinis Indonesia; dan (iii) panduan berbasis bukti untuk membangun sistem skrining DR otomatis pada layanan kesehatan primer dengan sumber daya terbatas.

### 1.2 Identifikasi Masalah

Berdasarkan latar belakang yang diuraikan pada Bagian 1.1, masalah penelitian dapat diidentifikasi sebagai berikut.

1. Beban *diabetic retinopathy* di Indonesia tinggi dan terus meningkat (insidens 34,6 per 1.000 orang-tahun; Sasongko et al., 2025), sementara rasio oftalmolog kurang dari dua spesialis per 100.000 penduduk dan terpusat di perkotaan Jawa, sehingga skrining fundus manual secara tahunan tidak layak secara struktural dan menuntut sistem interpretasi citra fundus otomatis pada layanan kesehatan primer.

2. Kualitas citra fundus yang dihasilkan perangkat berbiaya rendah di layanan primer sangat bervariasi (pencahayaan tidak merata, kontras rendah, dan *noise* akuisisi), dan penurunan kualitas ini menurunkan akurasi *grading* otomatis serta cenderung menyebabkan model menggrade pasien lebih rendah dari kondisi sebenarnya, yaitu jenis kesalahan yang berujung pada rujukan yang terlambat.

3. Pemilihan teknik *preprocessing* citra fundus pada literatur DR *grading* masih bersifat *ad-hoc*: tiap penelitian umumnya memilih satu teknik tanpa membandingkannya secara setara dengan teknik lain pada arsitektur dan protokol pelatihan yang sama, sehingga belum tersedia bukti komparatif *apple-to-apple* sebagai dasar pemilihan (Anupama et al., 2025).

4. Kinerja model DR *grading* menurun ketika diterapkan lintas-dataset akibat pergeseran domain (perbedaan perangkat kamera, populasi, dan protokol pencahayaan antarinstitusi), namun besarnya penurunan ini belum terkuantifikasi secara sistematis pada pasangan dataset yang berontologi label identik, padahal kondisi inilah yang akan dihadapi ketika sistem dioperasikan pada populasi Indonesia (Chokuwa dan Khan, 2025).

5. Belum tersedia panduan berbasis bukti mengenai kombinasi teknik *preprocessing* dan arsitektur *backbone* yang paling tangguh terhadap pergeseran domain untuk pengembangan sistem skrining DR otomatis pada layanan kesehatan primer dengan sumber daya terbatas di Indonesia.

### 1.3 Pembatasan Masalah

Agar penelitian terfokus dan layak dikerjakan dalam waktu penyusunan skripsi, serta untuk mengisolasi pengaruh *preprocessing* dan *backbone* dari faktor lain, masalah yang teridentifikasi pada Bagian 1.2 dibatasi sebagai berikut.

1. Tugas klasifikasi dibatasi pada *grading* lima kelas tingkat keparahan *diabetic retinopathy* pada skala ICDR untuk tingkat citra, tanpa segmentasi atau deteksi lesi.

2. Dataset yang digunakan hanya IDRiD *Disease Grading subset* (Porwal et al., 2020) untuk pelatihan dan evaluasi *in-distribution*, serta DDR (Li et al., 2019) untuk evaluasi lintas-dataset.

3. Teknik *preprocessing* yang dibandingkan dibatasi pada lima metode, yaitu CLAHE, Ben Graham *preprocessing*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*.

4. Arsitektur *backbone* yang dibandingkan dibatasi pada ResNet-50 (CNN) dan ViT-B/16 (*Vision Transformer*), keduanya dilatih melalui *transfer learning* dari bobot ImageNet tanpa *pretraining* domain-spesifik.

5. Evaluasi bersifat kuantitatif (akurasi, *quadratic-weighted kappa*, *macro-F1*) dengan visualisasi Grad-CAM pada konfigurasi terbaik. Evaluasi lintas-dataset bersifat *zero-shot transfer* tanpa adaptasi pada DDR. Studi klinisi (*reader study*) dan *deployment* prospektif berada di luar cakupan.

### 1.4 Rumusan Masalah

Berdasarkan identifikasi dan pembatasan masalah pada Bagian 1.2 dan 1.3, penelitian ini dirumuskan melalui tiga pertanyaan penelitian berikut.

1. Pada dataset IDRiD, bagaimana pengaruh lima teknik *preprocessing* (CLAHE, Ben Graham *preprocessing*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*) dan dua arsitektur *backbone* (ResNet-50 dan *Vision Transformer*) terhadap kinerja klasifikasi tingkat keparahan *diabetic retinopathy* yang diukur dengan akurasi, *quadratic-weighted kappa* (QWK), dan *macro-F1*?

2. Seberapa besar penurunan kinerja masing-masing konfigurasi *preprocessing* dan *backbone* ketika model yang dilatih pada IDRiD dievaluasi lintas-dataset pada DDR tanpa adaptasi, dan konfigurasi manakah yang menunjukkan ketahanan terbaik terhadap pergeseran domain antarinstitusi?

3. Bagaimana *trade-off* antara akurasi prediktif, ketahanan lintas-dataset, dan biaya komputasi (jumlah parameter dan waktu inferensi) dari setiap kombinasi *preprocessing* dan *backbone*, serta konfigurasi manakah yang paling sesuai untuk *deployment* sistem skrining DR pada layanan kesehatan primer di Indonesia?

### 1.5 Tujuan Penelitian

Sejalan dengan rumusan masalah pada Bagian 1.4, penelitian ini memiliki tiga tujuan sebagai berikut.

1. Menganalisis pengaruh lima teknik *preprocessing* (CLAHE, Ben Graham *preprocessing*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*) dan dua arsitektur *backbone* (ResNet-50 dan *Vision Transformer*) terhadap kinerja klasifikasi tingkat keparahan *diabetic retinopathy* pada dataset IDRiD, yang diukur melalui akurasi, *quadratic-weighted kappa* (QWK), dan *macro-F1*.

2. Mengkuantifikasi penurunan kinerja setiap konfigurasi *preprocessing* dan *backbone* ketika model yang dilatih pada IDRiD dievaluasi lintas-dataset pada DDR tanpa adaptasi, serta mengidentifikasi konfigurasi yang menunjukkan ketahanan terbaik terhadap pergeseran domain antarinstitusi.

3. Menganalisis *trade-off* antara akurasi prediktif, ketahanan lintas-dataset, dan biaya komputasi (jumlah parameter dan waktu inferensi) untuk memberikan rekomendasi konfigurasi yang paling sesuai untuk *deployment* sistem skrining *diabetic retinopathy* pada layanan kesehatan primer di Indonesia.

### 1.6 Manfaat Penelitian

Secara teoretis, penelitian ini menghasilkan perbandingan empiris yang *reproducible* antara teknik *preprocessing* dan arsitektur *backbone* pada DR *grading* di bawah protokol yang terunifikasi, sekaligus memperluas literatur dengan evaluasi lintas-dataset pada pasangan IDRiD dan DDR yang berontologi label identik namun berbeda perangkat, populasi, dan pencahayaan. Hasilnya menjadi bukti sejauh mana performa pada satu dataset DR *grading* dapat dipertahankan pada dataset lain, sekaligus menjawab langsung rekomendasi *future work* Anupama et al. (2025).

Secara praktis, penelitian ini memberikan panduan berbasis bukti bagi pengembangan sistem skrining DR otomatis pada layanan kesehatan primer di Indonesia, yang harus beroperasi pada perangkat dan populasi yang berbeda dari dataset pelatihan publik. Konfigurasi *preprocessing* dan *backbone* yang paling tangguh terhadap pergeseran domain dapat diadopsi sebagai titik awal pengembangan. Penelitian ini juga menghasilkan *pipeline* eksperimental terdokumentasi yang dapat digunakan ulang untuk menguji teknik *preprocessing*, *backbone*, atau dataset lain tanpa implementasi ulang.

## KAJIAN TEORI

### 2.1 Diabetic Retinopathy

#### 2.1.1 Patofisiologi Singkat

*Diabetic retinopathy* (DR) adalah komplikasi *microvascular* kronis dari *diabetes mellitus*. Hiperglikemia jangka panjang merusak kapiler retina melalui hilangnya sel *pericyte*, penebalan membran basal, dan kebocoran sawar darah-retina, yang berujung pada dua proses paralel, yaitu peningkatan permeabilitas kapiler (memicu *edema* dan penumpukan lipid) serta penyumbatan kapiler (memicu *ischaemia*); area iskemik kemudian memicu produksi *vascular endothelial growth factor* (VEGF) yang mendorong pembentukan pembuluh darah baru (Wong dan Sabanayagam, 2023). Secara klinis DR dibagi menjadi dua tahap. *Non-proliferative DR* (NPDR) ditandai oleh lesi seperti *microaneurysm*, perdarahan intraretinal, *hard exudate*, *cotton-wool spot*, *venous beading*, dan *intraretinal microvascular abnormalities* (IRMA), tanpa pembuluh darah baru. *Proliferative DR* (PDR) ditandai oleh *neovascularisation* yang berisiko menyebabkan perdarahan *vitreous* dan *tractional retinal detachment*. Jenis dan sebaran lesi inilah yang menjadi dasar penilaian tingkat keparahan pada skala ICDR (Subbab 2.1.3) sekaligus sasaran verifikasi visual Grad-CAM (Subbab 2.5.3).

#### 2.1.2 Beban Penyakit Global dan di Indonesia

DR merupakan penyebab utama kebutaan yang dapat dicegah pada populasi usia produktif. Teo et al. (2021) memproyeksikan jumlah penderita DR meningkat dari 103 juta (2020) menjadi sekitar 161 juta pada 2045, tren yang disebut Wong dan Sabanayagam (2023) sebagai "pandemi DR". Di Indonesia, beban ini tidak seimbang dengan ketersediaan tenaga oftalmologi: insidens DR mencapai 34,6 per 1.000 orang-tahun (Sasongko et al., 2025) dan prevalensi DR pada pasien di rumah sakit rujukan mencapai 55% (Saputra et al., 2024), sementara kepadatan dokter spesialis mata kurang dari dua per 100.000 penduduk dan terpusat di perkotaan Jawa. Kesenjangan inilah yang mendasari kebutuhan sistem skrining DR otomatis di layanan kesehatan primer (American Diabetes Association, 2024).

#### 2.1.3 Skala International Clinical Diabetic Retinopathy (ICDR)

Variabel target klinis pada penelitian ini adalah skala *International Clinical Diabetic Retinopathy* (ICDR) yang diusulkan Wilkinson et al. (2003) sebagai penyederhanaan skala ETDRS. ICDR membagi retinopati menjadi lima tingkat berurutan: *grade* 0 (tanpa retinopati), *grade* 1 (NPDR ringan, hanya *microaneurysm*), *grade* 2 (NPDR sedang, kondisi antara keduanya), *grade* 3 (NPDR berat, mengikuti aturan "4-2-1", yaitu perdarahan intraretinal pada empat kuadran, atau *venous beading* pada dua kuadran atau lebih, atau IRMA menonjol, tanpa *neovascularisation*), dan *grade* 4 (PDR, ditandai *neovascularisation* atau perdarahan *vitreous*/pre-retinal).

Skala ini bersifat ordinal: kelima kelas tersusun pada kontinum keparahan yang monoton, sehingga kesalahan klasifikasi berjarak dua *grade* lebih merugikan daripada satu *grade*. Sifat ini menuntut metrik yang sensitif terhadap urutan kelas seperti *quadratic-weighted kappa* (Subbab 2.5.2), bukan sekadar akurasi. Biaya klinis kesalahan pun asimetris: pasien *grade* 3 yang diprediksi *grade* 1 kehilangan rujukan yang seharusnya, sedangkan *grade* 0 yang diprediksi *grade* 1 memboroskan kapasitas spesialis.

### 2.2 Citra Fundus dan Benchmark Dataset

#### 2.2.1 Fotografi Fundus Berwarna

Fotografi fundus berwarna adalah modalitas pencitraan utama pada skrining DR karena non-invasif, relatif murah, dan tersedia luas di layanan primer. Sebuah citra fundus merekam *optic disc*, *macula*, *fovea*, dan *vascular arcade* pada sudut pandang 30 sampai 50 derajat. Akuisisi *non-mydriatic* (tanpa pelebaran pupil) lebih disukai pada skrining karena lebih nyaman dan tidak memerlukan supervisi klinis langsung. Kualitas citra dipengaruhi oleh *cataract*, ukuran pupil, dan keseragaman pencahayaan, sehingga sebagian citra dunia nyata tidak dapat dinilai (*ungradable*); dataset skrining modern umumnya menyertakan label kualitas untuk menyaringnya.

#### 2.2.2 Indian Diabetic Retinopathy Image Dataset (IDRiD)

*Indian Diabetic Retinopathy Image Dataset* (IDRiD; Porwal et al., 2020), *benchmark* resmi tantangan IEEE ISBI 2018, menjadi data *in-distribution* pada penelitian ini. IDRiD berisi 516 citra fundus dari satu klinik di Nanded, India, dengan satu jenis kamera (Kowa VX-10α). Penelitian ini hanya memakai subset *Disease Grading*, yang setiap citranya berlabel *grade* ICDR 0 sampai 4. Partisi resmi latih dan uji diadopsi tanpa modifikasi agar hasil sebanding dengan *benchmark* lain; rincian pembagian disajikan pada Subbab 3.2.

#### 2.2.3 Dataset for Diabetic Retinopathy (DDR)

*Dataset for Diabetic Retinopathy* (DDR; Li et al., 2019) digunakan sebagai target evaluasi lintas-dataset. DDR berisi 13.673 citra dari 147 rumah sakit di 23 provinsi Tiongkok dengan beragam jenis kamera, sehingga heterogenitas perangkat, kualitas, dan demografinya jauh melampaui IDRiD. Labelnya memakai skala ICDR 0 sampai 4 yang identik dengan IDRiD, ditambah label kualitas untuk memisahkan citra *gradable* dan *ungradable*. Penelitian ini hanya memakai partisi uji DDR setelah penyaringan citra *ungradable*; rincian disajikan pada Subbab 3.2.

#### 2.2.4 Rasionalisasi Pemasangan IDRiD dan DDR

Pemilihan pasangan IDRiD dan DDR didasari tiga pertimbangan. *Pertama*, keduanya berbagi ontologi label identik (skala ICDR lima kelas; Wilkinson et al., 2003), sehingga perubahan kinerja lintas-dataset dapat diatribusikan pada pergeseran domain (*domain shift*), bukan inkonsistensi anotasi. *Kedua*, keduanya berbeda pada dimensi yang relevan dengan *deployment*, yaitu negara, populasi, perangkat akuisisi (satu kamera satu lokasi versus banyak kamera di 147 lokasi), dan profil kualitas citra, sehingga model yang dilatih pada IDRiD dan diuji pada DDR menghadapi perkiraan realistis pergeseran distribusi antar sistem kesehatan. *Ketiga*, keduanya disertai *paper* deskriptor *peer-reviewed* (Porwal et al., 2020; Li et al., 2019) yang mendokumentasikan akuisisi, anotasi, dan partisi resmi, sehingga memenuhi syarat reproduktibilitas.

### 2.3 Teknik Preprocessing untuk Citra Fundus

Kualitas citra fundus mentah bervariasi akibat perbedaan pencahayaan, pewarnaan sensor, dan kondisi optik mata. *Preprocessing* bertujuan menyeragamkan karakteristik citra agar lesi klinis (*microaneurysm*, *haemorrhage*, *hard exudate*) lebih konsisten dikenali model. Penelitian ini mengevaluasi lima teknik berikut yang mewakili spektrum pendekatan pada literatur DR *grading*.

#### 2.3.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)

*Contrast Limited Adaptive Histogram Equalization* (CLAHE; Zuiderveld, 1994) menerapkan *histogram equalization* secara lokal pada *tile* kecil, sehingga mendistribusikan ulang intensitas piksel agar kontras meningkat di tiap wilayah. Untuk mencegah penguatan *noise*, tinggi *histogram* dibatasi oleh *clip limit*, dan kelebihan intensitas didistribusikan ulang secara merata. Pada citra fundus, CLAHE diterapkan pada kanal luminansi (ruang warna LAB) untuk mengungkap kontras lesi kecil seperti *microaneurysm* yang tersembunyi pada area berpencahayaan tidak merata.

#### 2.3.2 Ben Graham Normalization

Teknik *Ben Graham Normalization*, yang diperkenalkan pemenang kompetisi Kaggle *Diabetic Retinopathy Detection* (2015) dan menjadi *preprocessing* baku pada banyak studi DR *grading*, mengurangi komponen pencahayaan berskala besar dengan mengurangkan versi *Gaussian-blurred* dari citra asli, $I_{\mathrm{norm}} = \alpha I + \beta G_{\sigma}(I) + \gamma$, dengan $G_{\sigma}$ adalah *filter Gaussian* beradius $\sigma$. Akibatnya detail berskala kecil seperti lesi dan pembuluh darah lebih menonjol, sementara variasi pencahayaan global antar kamera berkurang.

#### 2.3.3 Adaptive Sigmoid Enhancement

*Adaptive Sigmoid Enhancement* meregangkan rentang intensitas dengan fungsi *sigmoid* $f(x) = 1 / (1 + \exp(-\alpha (x - \beta)))$, dengan $\alpha$ mengatur kemiringan dan $\beta$ titik tengah. Ketika $\alpha$ dan $\beta$ diambil dari rata-rata dan standar deviasi intensitas lokal, kontras diperkuat di area gelap tanpa memperbesar *noise* di area terang, sehingga lesi pucat lebih mudah dibedakan dari latar retina.

#### 2.3.4 LAB Adaptive Contrast Enhancement (LAB-ACE)

*LAB Adaptive Contrast Enhancement* (LAB-ACE) memisahkan luminansi (L) dari komponen warna (A dan B) pada ruang warna LAB, lalu meningkatkan kontras hanya pada kanal L (umumnya melalui CLAHE dan normalisasi lokal) dan merekonstruksinya kembali ke RGB. Dengan mempertahankan kanal A dan B, teknik ini memperbaiki kontras lesi tanpa distorsi warna yang dapat mengacaukan interpretasi.

#### 2.3.5 Multi-channel Image Enhancement

*Multi-channel Image Enhancement* (MCIE) menggabungkan beberapa representasi citra menjadi satu masukan tiga kanal, misalnya kanal hijau asli (peka terhadap *haemoglobin*), hasil CLAHE pada kanal L, dan hasil *Ben Graham Normalization*. Dengan demikian *feature extractor* memanfaatkan beberapa *enhancement* komplementer sekaligus, tidak bergantung pada satu jenis saja. Anupama et al. (2025) mengeksplorasi kombinasi semacam ini dan merekomendasikan kajian sistematis terhadap variasi *preprocessing* sebagai arah lanjutan.

### 2.4 Arsitektur Deep Learning untuk Klasifikasi Citra

Penelitian ini mengevaluasi dua *backbone deep learning* yang mewakili dua paradigma berbeda, yaitu *convolutional neural network* (CNN) yang diwakili ResNet-50 dan *Vision Transformer* (ViT) yang diwakili ViT-B/16. Arsitektur masing-masing diuraikan pada Subbab 2.4.1 dan 2.4.2.

#### 2.4.1 Convolutional Neural Network dan ResNet-50

*Convolutional neural network* (CNN) memetakan citra ke keluaran melalui konvolusi, aktivasi non-linear, dan *pooling* yang berlapis. Operasi konvolusi menerapkan *kernel* kecil pada tiap posisi spasial, sehingga jumlah parameter tidak bergantung pada ukuran citra (*weight sharing*) dan model bersifat *translation-equivariant*. Penumpukan konvolusi, aktivasi seperti $\mathrm{ReLU}(z) = \max(0, z)$, dan *down-sampling* menghasilkan hirarki *receptive field*, dari tepi dan tekstur pada lapisan awal hingga objek utuh pada lapisan dalam. Untuk klasifikasi $K$ kelas, vektor fitur akhir dipetakan ke *logit* $z \in \mathbb{R}^{K}$, diubah menjadi probabilitas melalui *softmax* $p_k = \exp(z_k) / \sum_{j=1}^{K} \exp(z_j)$, dan dilatih dengan *categorical cross-entropy* $\mathcal{L}_{\mathrm{CE}} = -\sum_{k=1}^{K} y_k \log p_k$.

ResNet-50 (He et al., 2016) adalah salah satu CNN yang paling banyak dipakai pada pencitraan medis. Kontribusi utamanya adalah *residual connection* $y = F(x) + x$, yaitu jalur pintas yang mengatasi *vanishing gradient* dan memungkinkan pelatihan jaringan dalam. ResNet-50 menyusun 50 lapisan konvolusi (unit penyusunnya blok *bottleneck* tiga lapis dengan *residual connection*) dalam empat *stage* dengan resolusi spasial menurun, sebagaimana disajikan pada Gambar 2.1. Arsitektur ini dipilih sebagai *backbone* CNN karena mapan, berukuran parameter moderat (~25,5 juta), dan lazim menjadi *baseline* pada literatur DR *grading*.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gambar/resnet.png}
\caption{Arsitektur ResNet-50 (He et al., 2016) dengan citra fundus sebagai masukan; \emph{inset} menampilkan blok \emph{bottleneck residual} ($y = F(x) + x$). Kepala klasifikasi ImageNet diganti menjadi lima kelas ICDR pada penelitian ini.}
\label{fig:resnet}
\end{figure}

#### 2.4.2 Vision Transformer (ViT)

*Vision Transformer* (ViT; Dosovitskiy et al., 2021) mengadopsi arsitektur *Transformer* (Vaswani et al., 2017) ke pengenalan citra. ViT membagi citra menjadi *patch* berukuran tetap ($16 \times 16$ piksel pada ViT-B/16), melinearisasi tiap *patch* menjadi *token*, menambahkan *positional embedding*, lalu memprosesnya dengan beberapa blok *Transformer encoder*. Komponen intinya adalah *self-attention*, yang membuat setiap *token* menimbang relevansi seluruh *token* lain melalui proyeksi *query* $Q$, *key* $K$, dan *value* $V$: $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^{\top}/\sqrt{d_k}) V$. Kemampuan memodelkan hubungan jangka jauh (*long-range dependencies*) ini relevan untuk DR *grading* karena lesi dapat tersebar di berbagai kuadran retina. Arsitektur ViT-B/16 beserta struktur blok *Transformer encoder* dan mekanisme *multi-head self-attention*-nya disajikan pada Gambar 2.2. Anupama et al. (2025) secara eksplisit mengusulkan ViT sebagai arah pengembangan untuk DR *grading*.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gambar/vit-pas.png}
\caption{Arsitektur ViT-B/16 (Dosovitskiy et al., 2021; Vaswani et al., 2017) dengan citra fundus sebagai masukan; \emph{inset} menampilkan blok \emph{Transformer encoder} dan mekanisme \emph{multi-head self-attention}. Kepala klasifikasi ImageNet diganti menjadi lima kelas ICDR pada penelitian ini.}
\label{fig:vit}
\end{figure}

#### 2.4.3 Transfer Learning

Dataset DR *grading* publik relatif kecil (IDRiD hanya memiliki beberapa ratus citra latih), sehingga melatih ResNet-50 atau ViT dari inisialisasi acak hampir pasti *overfitting*. *Transfer learning* mengatasinya dengan menginisialisasi *backbone* dari bobot hasil *pretraining* pada dataset besar (ImageNet), yang telah mempelajari fitur visual umum seperti tepi, tekstur, dan bentuk, lalu melakukan *fine-tuning* pada dataset target dengan *classifier head* yang diganti sesuai jumlah kelas. Pendekatan ini mengurangi kebutuhan data dan mempercepat konvergensi, sehingga pelatihan kedua *backbone* pada IDRiD menjadi layak secara komputasional.

### 2.5 Metrik Evaluasi Klasifikasi DR Grading

#### 2.5.1 Akurasi, Presisi, Recall, dan Macro-F1

Akurasi adalah proporsi prediksi benar, $\mathrm{Akurasi} = (1/N) \sum_{i=1}^{N} \mathbb{1}[\hat{y}_{i} = y_{i}]$, namun bias pada data tak seimbang yang didominasi *grade* 0 seperti IDRiD dan DDR. Karena itu dilaporkan pula presisi dan *recall* per kelas beserta *macro-F1*, yaitu rata-rata aritmatik dari $F1 = 2\,(\mathrm{presisi} \cdot \mathrm{recall}) / (\mathrm{presisi} + \mathrm{recall})$ atas seluruh kelas. *Macro-F1* memberi bobot setara pada tiap kelas, sehingga sensitif terhadap kinerja pada kelas minoritas.

#### 2.5.2 Quadratic-weighted Kappa (QWK)

Karena ICDR ordinal, kesalahan berjarak dua *grade* lebih parah daripada satu *grade*, padahal akurasi dan *F1* tidak membedakannya. *Quadratic-weighted kappa* (Cohen, 1968) memberi penalti sebanding dengan kuadrat jarak antar-kelas,

$$
\kappa_{w} \;=\; 1 - \frac{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} O_{ij}}{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} E_{ij}},
\qquad w_{ij} = \frac{(i - j)^{2}}{(K - 1)^{2}},
$$

dengan $O$ sebagai *confusion matrix* observasi dan $E$ sebagai *confusion matrix* harapan jika prediksi dan label independen. Nilai $\kappa_w = 1$ berarti prediksi sempurna dan $\kappa_w = 0$ setara tebakan acak. QWK adalah metrik standar pada tantangan DR *grading* (Kaggle *Diabetic Retinopathy Detection*, APTOS 2019) dan menjadi metrik utama penelitian ini.

#### 2.5.3 Gradient-weighted Class Activation Mapping (Grad-CAM)

*Gradient-weighted Class Activation Mapping* (Grad-CAM; Selvaraju et al., 2017) menghasilkan *heatmap* yang menandai wilayah citra paling berpengaruh terhadap keputusan klasifikasi, dengan membobot *feature map* lapisan konvolusi terakhir memakai gradien skor kelas target. Pada DR *grading*, Grad-CAM dipakai untuk memverifikasi apakah model mendasarkan prediksi pada lesi yang relevan secara klinis (misalnya *microaneurysm* atau *haemorrhage*) atau pada artefak seperti tepi lensa. Pada penelitian ini, Grad-CAM berperan sebagai alat interpretasi tambahan pada konfigurasi terbaik, bukan fokus kuantitatif utama, sejalan dengan rekomendasi Anupama et al. (2025).

### 2.6 Penelitian Terkait

Era modern klasifikasi DR berbasis *deep learning* dimulai dari Gulshan et al. (2016), yang melatih CNN pada 128.175 citra retina dan melaporkan sensitivitas 90,3% serta spesifisitas 98,1% untuk DR yang perlu rujukan, setara ahli. Abramoff et al. (2018) membawanya ke ranah regulasi melalui IDx-DR, sistem AI pertama yang disetujui FDA untuk keputusan diagnostik di layanan primer tanpa supervisi spesialis. Keduanya menunjukkan DR *grading* otomatis layak secara teknis, bermanfaat klinis, dan dapat disetujui regulator.

Rilis dataset publik seperti IDRiD (Porwal et al., 2020) dan DDR (Li et al., 2019) menggeser fokus ke *benchmark* terbuka, tempat berbagai *backbone* (ResNet, DenseNet, Inception, EfficientNet) dievaluasi. Zhou et al. (2023) mengembangkan RETFound, *foundation model* berbasis ViT yang dilatih secara *self-supervised* pada 1,6 juta citra retina dan konsisten lebih unggul daripada bobot ImageNet pada tugas retina, terutama saat data terbatas.

Chokuwa dan Khan (2025) menunjukkan model DR *grading* mengalami penurunan performa yang nyata ketika dievaluasi pada dataset dengan pengaturan akuisisi berbeda, akibat pergeseran domain (kamera, pencahayaan, demografi), bukan inkonsistensi anotasi. Temuan ini mendasari pilihan penelitian ini untuk menjadikan evaluasi lintas-dataset sebagai ukuran kinerja yang lebih realistis.

*Paper* anchor penelitian ini adalah Anupama et al. (2025) di *Scientific Reports*, yang mengevaluasi beberapa *backbone* untuk DR *grading* pada dataset tunggal dan, pada bagian *future work*, mengidentifikasi tiga arah: (i) eksplorasi sistematis kombinasi *preprocessing*; (ii) validasi pada *multi-center datasets* dengan perangkat dan demografi beragam; serta (iii) integrasi alat interpretasi seperti Grad-CAM. Penelitian ini mengambil ketiganya sebagai landasan, yaitu membandingkan lima teknik *preprocessing* (Subbab 2.3) pada dua *backbone* berbeda paradigma (Subbab 2.4), mengevaluasi lintas-dataset IDRiD dan DDR (Subbab 2.2), serta memakai Grad-CAM pada konfigurasi terbaik (Subbab 2.5.3).

### 2.7 Kerangka Berpikir

Citra fundus mentah memiliki variasi pencahayaan, kontras, dan pewarnaan sensor (Subbab 2.2.1), sedangkan lesi awal *diabetic retinopathy* seperti *microaneurysm* berukuran kecil dan berkontras rendah terhadap latar retina (Subbab 2.1.1). Kelima teknik *preprocessing* (Subbab 2.3) bekerja dengan mekanisme yang berbeda: CLAHE dan LAB-ACE memperkuat kontras lokal, Ben Graham *Normalization* menstabilkan pencahayaan global antar-kamera, *Adaptive Sigmoid Enhancement* meregangkan rentang intensitas secara adaptif, dan *Multi-channel Image Enhancement* menggabungkan beberapa representasi sekaligus. Karena mekanismenya berbeda, dampak tiap teknik terhadap visibilitas lesi, proses ekstraksi fitur, dan kinerja klasifikasi diperkirakan juga berbeda, sehingga pemilihan teknik *preprocessing* bukan hal yang netral terhadap kinerja akhir.

Pada sisi arsitektur, kedua *backbone* mewakili dua paradigma dengan *inductive bias* yang berbeda (Subbab 2.4). ResNet-50 sebagai *convolutional neural network* menangkap fitur lokal melalui *receptive field* berlapis, sehingga peka terhadap lesi kecil yang terlokalisasi, sedangkan ViT-B/16 sebagai *Vision Transformer* memodelkan hubungan jangka jauh antar-wilayah melalui *self-attention*, yang relevan karena lesi DR dapat tersebar di berbagai kuadran retina dan menentukan tingkat keparahan pada skala ICDR (Subbab 2.1.3). Perbedaan *inductive bias* ini diperkirakan menghasilkan kinerja dan ketahanan terhadap pergeseran domain yang berbeda pula.

Ketika model yang dilatih pada satu sumber data (IDRiD) diuji pada sumber lain (DDR), model menghadapi pergeseran domain akibat perbedaan kamera, populasi, dan pencahayaan, yang menurunkan kinerja meskipun ontologi label identik (Subbab 2.6; Chokuwa dan Khan, 2025). Besar penurunan ini diperkirakan bergantung pada seberapa baik *preprocessing* menyeragamkan tampilan citra antar-sumber dan seberapa tangguh *backbone* terhadap variasi yang tersisa. Dengan demikian, kinerja klasifikasi DR *grading* dipandang sebagai fungsi dari interaksi faktor *preprocessing* dan faktor *backbone*. Desain faktorial $5 \times 2$ pada penelitian ini (Subbab 3.1) ditujukan untuk mengisolasi pengaruh kedua faktor tersebut, sementara sifat ordinal ICDR dan ketidakseimbangan kelas menuntut evaluasi dengan QWK dan *macro-F1*, bukan akurasi semata (Subbab 2.5). Kerangka berpikir inilah yang mendasari hipotesis pada Subbab 2.8.

### 2.8 Hipotesis Penelitian

Berdasarkan kerangka berpikir di atas, dirumuskan tiga hipotesis penelitian yang sejalan dengan ketiga rumusan masalah. Hipotesis ini bersifat terarah dan akan diuji secara empiris; signifikansi perbedaan antar-konfigurasi dinilai melalui interval kepercayaan berbasis *bootstrap* (Subbab 3.7).

1. **H1.** Terdapat perbedaan kinerja klasifikasi (akurasi, *quadratic-weighted kappa*, dan *macro-F1*) yang bermakna antar lima teknik *preprocessing* dan antar dua arsitektur *backbone* pada dataset IDRiD. Secara terarah, teknik yang menstabilkan pencahayaan global dan menggabungkan beberapa representasi (Ben Graham *Normalization* dan *Multi-channel Image Enhancement*) diperkirakan mengungguli teknik kontras-lokal tunggal, dan ViT-B/16 diperkirakan setidaknya menyamai ResNet-50 karena kemampuannya memodelkan lesi yang tersebar.

2. **H2.** Seluruh konfigurasi mengalami penurunan kinerja ketika dievaluasi lintas-dataset pada DDR akibat pergeseran domain, tetapi besar penurunannya berbeda antar-konfigurasi. Konfigurasi dengan *preprocessing* yang menstandarkan pencahayaan dan warna (Ben Graham *Normalization*, LAB-ACE, dan *Multi-channel Image Enhancement*) yang dipasangkan dengan *backbone* ViT-B/16 diperkirakan mengalami penurunan terkecil, sehingga paling tahan terhadap pergeseran domain antarinstitusi.

3. **H3.** Terdapat *trade-off* antara akurasi prediktif, ketahanan lintas-dataset, dan biaya komputasi: konfigurasi dengan kinerja tertinggi pada IDRiD belum tentu paling tahan lintas-dataset maupun paling efisien. Konfigurasi yang menyeimbangkan ketiga aspek tersebut diperkirakan paling sesuai untuk *deployment* sistem skrining DR pada layanan kesehatan primer di Indonesia.

## METODE PENELITIAN

### 3.1 Desain Penelitian

Penelitian ini menggunakan desain eksperimen kuantitatif berbasis kerangka analisis komparatif faktorial. Variabel independen terdiri atas dua faktor, yaitu teknik *preprocessing* citra fundus pada lima tingkat (CLAHE, Ben Graham *Normalization*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*) dan arsitektur *backbone deep learning* pada dua tingkat (ResNet-50 dan ViT-B/16). Kombinasi kedua faktor menghasilkan sepuluh konfigurasi eksperimen. Definisi operasional seluruh variabel diuraikan pada Subbab 3.3. Setiap konfigurasi dilatih pada dataset IDRiD menggunakan protokol pelatihan yang identik, kemudian dievaluasi pada dua partisi uji yang berbeda, yaitu partisi uji IDRiD (*in-distribution*) dan partisi uji DDR (lintas-dataset). Perbandingan antar-konfigurasi dilakukan melalui metrik akurasi, *quadratic-weighted kappa* (QWK), *macro-F1*, serta *confusion matrix* per kelas.

Alur eksperimen terdiri atas enam tahap berurutan, yaitu akuisisi dan pembagian dataset, penerapan salah satu dari lima teknik *preprocessing* pada seluruh citra (Subbab 3.4), inisialisasi *backbone* ResNet-50 atau ViT-B/16 dari bobot ImageNet (Subbab 3.5), pelatihan model dengan *random seed* yang tetap (Subbab 3.6), evaluasi pada IDRiD *test* dan DDR *test* (Subbab 3.7), serta analisis statistik dan visualisasi Grad-CAM pada konfigurasi terbaik. Seluruh *hyperparameter* pelatihan dan *pipeline* augmentasi bersifat identik antar-konfigurasi, sehingga perbedaan kinerja yang teramati dapat diatribusikan pada teknik *preprocessing* dan pilihan *backbone*, bukan pada faktor perancangan lain. Bobot model, *seed*, skrip, dan log metrik disimpan pada repositori Git publik untuk menjamin reproduktibilitas. Gambar 3.1 menyajikan alur eksperimen secara keseluruhan.

\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
  font=\small,
  ttl/.style={font=\itshape, align=center},
  item/.style={draw, rectangle, fill=white, align=center, font=\scriptsize, inner sep=3pt, minimum height=0.55cm, text width=3.3cm},
  grp/.style={draw, rectangle, fill=gray!4, inner sep=4pt},
  db/.style={draw, cylinder, shape border rotate=90, aspect=0.28, fill=gray!8, align=center, font=\scriptsize, minimum width=2.0cm, minimum height=1.5cm},
  arr/.style={-Stealth, thick}
]
\node[db] (data) at (0,-1.7) {Dataset\\\emph{IDRiD}\\$+$ \emph{DDR}};

\node[ttl] (prep_t) at (4.7,0) {Preprocessing};
\node[item, below=1.4mm of prep_t] (prep1) {Cropping $+$ Resize $224{\times}224$};
\node[item, below=1mm of prep1] (prep2) {5 teknik \emph{enhancement} (Faktor 1)};
\begin{scope}[on background layer]\node[grp, fit=(prep_t)(prep1)(prep2)] (prep) {};\end{scope}

\node[ttl] (aug_t) at (4.7,-3.6) {Augmentasi};
\node[item, below=1.4mm of aug_t] (aug1) {\emph{Flip}, rotasi $\pm 30^{\circ}$, \emph{brightness/contrast jitter}};
\begin{scope}[on background layer]\node[grp, fit=(aug_t)(aug1)] (aug) {};\end{scope}

\node[ttl] (back_t) at (9.4,0) {Backbone (Faktor 2)};
\node[item, below=1.4mm of back_t] (back1) {ResNet-50 (ImageNet-1k)};
\node[item, below=1mm of back1] (back2) {ViT-B/16 (ImageNet-21k)};
\begin{scope}[on background layer]\node[grp, fit=(back_t)(back1)(back2)] (back) {};\end{scope}

\node[ttl] (pel_t) at (9.4,-3.6) {Pelatihan};
\node[item, below=1.4mm of pel_t] (pel1) {$5 \times 2 = 10$ konfigurasi};
\node[item, below=1mm of pel1] (pel2) {AdamW, \emph{class-weighted} CE};
\begin{scope}[on background layer]\node[grp, fit=(pel_t)(pel1)(pel2)] (pel) {};\end{scope}

\node[ttl] (eval_t) at (14.1,0) {Evaluasi};
\node[item, below=1.4mm of eval_t] (eval1) {IDRiD \emph{test} (\emph{in-distribution})};
\node[item, below=1mm of eval1] (eval2) {DDR \emph{test} (lintas-\emph{dataset}, \emph{zero-shot})};
\begin{scope}[on background layer]\node[grp, fit=(eval_t)(eval1)(eval2)] (eval) {};\end{scope}

\node[ttl] (metr_t) at (14.1,-3.6) {Metrik \& Interpretasi};
\node[item, below=1.4mm of metr_t] (metr1) {Akurasi, QWK, \emph{Macro-F1}};
\node[item, below=1mm of metr1] (metr2) {Grad-CAM (konfigurasi terbaik)};
\begin{scope}[on background layer]\node[grp, fit=(metr_t)(metr1)(metr2)] (metr) {};\end{scope}

\draw[arr] (data.east) -- ++(0.4,0) |- (prep.west);
\draw[arr] (prep.south) -- (aug.north);
\draw[arr] (aug.east) -- ++(0.4,0) |- (back.west);
\draw[arr] (back.south) -- (pel.north);
\draw[arr] (pel.east) -- ++(0.4,0) |- (eval.west);
\draw[arr] (eval.south) -- (metr.north);
\end{tikzpicture}%
}
\caption{Alur penelitian: kombinasi lima teknik \emph{preprocessing} (Faktor 1) dan dua \emph{backbone} (Faktor 2) membentuk sepuluh konfigurasi ($5 \times 2$) yang dievaluasi pada IDRiD \emph{test} (\emph{in-distribution}) dan DDR \emph{test} (lintas-\emph{dataset}).}
\label{fig:pipeline}
\end{figure}

### 3.2 Dataset

#### 3.2.1 IDRiD Disease Grading Subset

Dari partisi resmi IDRiD (413 citra latih, 103 citra uji; Porwal et al., 2020), penelitian ini menyisihkan 15% citra latih (62 citra) sebagai partisi validasi melalui pengambilan acak bertingkat (*stratified*) di bawah *seed* tetap, menyisakan 351 citra latih. Partisi validasi dipakai untuk *early stopping* dan pemilihan *checkpoint*, sedangkan partisi uji 103 citra hanya diakses sekali pada akhir eksperimen untuk menghasilkan metrik *in-distribution*.

#### 3.2.2 DDR untuk Evaluasi Lintas-Dataset

DDR (Li et al., 2019) diperlakukan sebagai *set* evaluasi *zero-shot transfer*: hanya partisi uji resmi yang dipakai, setelah citra *ungradable* disaring agar ruang label tetap lima kelas, menghasilkan sekitar 3.759 citra. Partisi latih dan validasi DDR tidak digunakan sama sekali.

#### 3.2.3 Distribusi Kelas dan Penanganan Ketidakseimbangan

Kedua dataset berdistribusi tidak seimbang (*long-tailed*): *grade* 0 dan 2 mendominasi, sedangkan *grade* 1, 3, dan 4 minoritas. Ketidakseimbangan ditangani pada dua level, yaitu (i) *weighted random sampler* dengan bobot $\propto 1/\sqrt{n_k}$ ($n_k$ adalah jumlah citra latih kelas $k$) pada tiap *mini-batch*, dan (ii) *class-weighted categorical cross-entropy* dengan bobot $\propto 1/\sqrt{n_k}$ yang dinormalisasi agar berjumlah $K$. QWK dipilih sebagai metrik pemilihan *checkpoint* karena sensitif terhadap jarak ordinal dan relatif tahan terhadap distribusi marginal.

### 3.3 Variabel dan Definisi Operasional Variabel

Penelitian ini melibatkan dua variabel bebas, satu kelompok variabel terikat, dan sejumlah variabel kontrol. Definisi operasional masing-masing variabel diuraikan sebagai berikut.

**Variabel bebas.** Variabel bebas pertama adalah *teknik preprocessing* citra fundus, yaitu variabel kategorikal dengan lima taraf (CLAHE, Ben Graham *Normalization*, *Adaptive Sigmoid Enhancement*, LAB-ACE, dan *Multi-channel Image Enhancement*); secara operasional, teknik ini adalah transformasi intensitas atau warna yang diterapkan pada setiap citra setelah *cropping* dan sebelum *resize*, dengan parameter yang dirinci pada Subbab 3.4. Variabel bebas kedua adalah *arsitektur backbone*, yaitu variabel kategorikal dengan dua taraf (ResNet-50 dan ViT-B/16); secara operasional, *backbone* adalah jaringan ekstraksi fitur yang diinisialisasi dari bobot ImageNet lalu di-*fine-tune* pada data DR *grading*, sebagaimana dirinci pada Subbab 3.5. Kombinasi kedua variabel bebas membentuk sepuluh konfigurasi eksperimen ($5 \times 2$).

**Variabel terikat.** Variabel terikat utama adalah kinerja klasifikasi tingkat keparahan DR, yang secara operasional diukur dengan tiga metrik, yaitu akurasi, *quadratic-weighted kappa* (QWK), dan *macro-F1* (rumus pada Subbab 2.5), dihitung pada partisi uji. Selain itu diturunkan dua variabel terikat pelengkap, yaitu (a) penurunan kinerja lintas-dataset, didefinisikan sebagai selisih tiap metrik antara IDRiD *test* dan DDR *test* (Subbab 3.7) sebagai ukuran ketahanan terhadap pergeseran domain, dan (b) biaya komputasi, didefinisikan sebagai jumlah parameter model dan waktu inferensi rata-rata per citra.

**Variabel kontrol.** Agar perbedaan kinerja yang teramati dapat diatribusikan murni pada kedua variabel bebas, faktor-faktor berikut dibuat identik antar-konfigurasi: *pipeline* dasar *preprocessing* (Subbab 3.4.1), protokol augmentasi (Subbab 3.4.3), *optimiser* dan *hyperparameter* pelatihan (Subbab 3.6), partisi data dan *random seed*, serta perangkat dan lingkungan komputasi (Subbab 3.8).

### 3.4 Preprocessing Data

#### 3.4.1 Pipeline Dasar

Sebelum teknik spesifik diterapkan, setiap citra IDRiD dan DDR melewati *pipeline* dasar yang identik, yaitu (i) estimasi *mask* retina via *threshold* adaptif pada kanal hijau lalu *cropping* ke *bounding box* minimum untuk membuang bingkai hitam; (ii) *resize* ke $224 \times 224$ piksel (interpolasi *bilinear*) sesuai masukan baku bobot ImageNet; dan (iii) normalisasi ke $[0, 1]$ diikuti normalisasi per kanal dengan rata-rata dan standar deviasi ImageNet ($\mu = (0{,}485, 0{,}456, 0{,}406)$, $\sigma = (0{,}229, 0{,}224, 0{,}225)$). *Pipeline* ini identik pada pelatihan, validasi, uji, dan antar-dataset, agar pergeseran domain antara IDRiD dan DDR tidak tersamarkan oleh perbedaan *preprocessing*.

#### 3.4.2 Penerapan Lima Teknik Preprocessing

Teknik *preprocessing* diterapkan antara *cropping* dan *resize*; parameter tiap teknik (konsep pada Subbab 2.3) sebagai berikut. **CLAHE**: *clip limit* 2,0 dan *tile* $8 \times 8$ pada kanal L (LAB). **Ben Graham Normalization**: $\alpha = 4$, radius *Gaussian* $\sigma$ sebesar 10% diameter retina, dan $\gamma = 128$, mengikuti resep asli Kaggle 2015. **Adaptive Sigmoid Enhancement**: $\alpha$ dan $\beta$ dihitung adaptif dari rata-rata dan standar deviasi intensitas *patch* lokal. **LAB-ACE**: CLAHE pada kanal L lalu rekonstruksi ke RGB. **MCIE**: gabungan kanal hijau asli, hasil CLAHE pada kanal L, dan hasil *Ben Graham Normalization* menjadi citra tiga kanal.

#### 3.4.3 Augmentasi Data

Augmentasi daring diterapkan hanya pada partisi pelatihan IDRiD, setelah *preprocessing*. Augmentasi geometri berupa *horizontal* dan *vertical flip* (probabilitas 0,5) serta rotasi acak $[-30^{\circ}, +30^{\circ}]$, yang mempertahankan label karena *grade* ICDR tidak bergantung pada orientasi spasial. Augmentasi fotometri berupa *brightness* dan *contrast jitter* acak $[-0{,}2, +0{,}2]$ untuk mensimulasikan variasi pencahayaan antar-lokasi. Partisi validasi dan uji (IDRiD maupun DDR) tidak diaugmentasi.

### 3.5 Arsitektur Model

Konsep kedua *backbone* beserta diagram arsitekturnya telah diuraikan pada Subbab 2.4 (Gambar 2.1 dan 2.2); bagian ini hanya merinci konfigurasi eksperimennya. ResNet-50 dimuat dengan bobot *pretrained* ImageNet-1k melalui *torchvision* 0.18, sedangkan ViT-B/16 dengan bobot *pretrained* ImageNet-21k melalui *timm* 1.0. Pada kedua *backbone*, *classification head* asli diganti dengan *linear layer* baru menuju lima *logit* ICDR (ResNet-50: $2048 \to 5$ dari vektor *global average pooling*; ViT-B/16: $768 \to 5$ dari *token* CLS). Seluruh parameter dilatih bersama (*full fine-tuning*) tanpa pembekuan lapisan, dengan keluaran *softmax* yang dioptimasi memakai *class-weighted categorical cross-entropy* (Subbab 3.2.3 dan 3.6).

### 3.6 Protokol Pelatihan

Seluruh model dilatih dengan *optimiser* AdamW (Loshchilov dan Hutter, 2019), *learning rate* awal $\eta_{0} = 1 \times 10^{-4}$, *weight decay* $1 \times 10^{-4}$, $\beta_{1} = 0{,}9$, dan $\beta_{2} = 0{,}999$. *Learning rate* mengikuti jadwal *cosine* hingga $\eta_{\min} = 1 \times 10^{-6}$, didahului *warm-up* linear tiga *epoch*. Ukuran *batch* 16 citra, sesuai memori satu GPU NVIDIA T4 pada Google Colab Pro. Pelatihan berlangsung hingga 50 *epoch* dengan *early stopping* berbasis QWK validasi (kesabaran 10 *epoch*); *checkpoint* ber-QWK validasi tertinggi dipakai untuk evaluasi uji.

Fungsi kerugian adalah *class-weighted categorical cross-entropy* (bobot sesuai Subbab 3.2.3). *Mixup*, *label smoothing*, dan *focal loss* sengaja tidak dipakai agar perbedaan kinerja antar-konfigurasi murni berasal dari teknik *preprocessing* dan pilihan *backbone*. Seluruh *random number generator* diinisialisasi dengan *seed* yang sama agar hasil dapat direproduksi. Pelatihan satu konfigurasi memakan waktu sekitar 2 jam pada GPU T4, sehingga sepuluh konfigurasi membutuhkan sekitar 20 jam.

### 3.7 Protokol Evaluasi

Partisi uji IDRiD (103 citra) dievaluasi sekali per konfigurasi dengan akurasi, QWK, *macro-F1*, dan *confusion matrix* per kelas. Partisi uji DDR (setelah penyaringan *ungradable*) dievaluasi dengan *metric suite* identik pada *setting zero-shot transfer*, yaitu tanpa *fine-tuning*, *batch-normalisation recalibration*, atau adaptasi *threshold*. Penurunan tiap metrik pada DDR relatif terhadap IDRiD mencerminkan ketahanan domain konfigurasi.

Setiap metrik dilaporkan dengan interval kepercayaan 95% via *bootstrap* ($B = 1.000$ *resample*); perbandingan antar-konfigurasi memakai *paired bootstrap* dengan indeks *resample* yang sama agar variabilitas tingkat sampel terkendali. Dua konfigurasi dianggap berbeda bermakna bila interval kepercayaannya tidak tumpang tindih. Selain itu, Grad-CAM (Selvaraju et al., 2017) dihasilkan pada konfigurasi terbaik tiap *backbone* untuk verifikasi visual dasar prediksi.

### 3.8 Alat dan Lingkungan Implementasi

*Pipeline* eksperimen diimplementasikan dalam Python 3.11 menggunakan pustaka berikut: PyTorch 2.3 sebagai *backend tensor* dan *autodiff*; *torchvision* 0.18 untuk ResNet-50 dan bobot *pretrained* ImageNet-1k; *timm* 1.0 (Wightman, 2019) untuk ViT-B/16 dan bobot *pretrained* ImageNet-21k; OpenCV 4.9 dan *scikit-image* 0.22 untuk implementasi lima teknik *preprocessing*; Albumentations 1.4 untuk *pipeline* augmentasi; *scikit-learn* 1.4 untuk implementasi silang metrik evaluasi; serta Matplotlib 3.8 dan seaborn 0.13 untuk visualisasi matriks kebingungan dan *heatmap* Grad-CAM. Eksperimen dijalankan pada Google Colab Pro dengan GPU NVIDIA T4 atau A100 sesuai ketersediaan. Seluruh kode, skrip pelatihan, skrip evaluasi, berkas konfigurasi, dan log metrik disimpan pada repositori Git publik yang disertai *README* reproduksi dengan versi pustaka, *seed*, dan perintah eksekusi yang eksplisit.

### 3.9 Jadwal Penelitian

Kegiatan penelitian direncanakan berlangsung selama dua belas bulan, yaitu dari Maret 2026 sampai Februari 2027, mencakup studi literatur dan penyusunan proposal, seminar proposal dan revisinya, persiapan lingkungan dan pengumpulan data, implementasi *pipeline preprocessing*, pelatihan sepuluh konfigurasi model, evaluasi *in-distribution* dan lintas-*dataset*, analisis statistik dan visualisasi Grad-CAM, penulisan BAB IV dan BAB V, seminar hasil, serta ujian skripsi dan pengumpulan skripsi final. Rincian jadwal bulanan disajikan pada Tabel 3.1; kolom bulan Maret sampai Desember merupakan bulan pada tahun 2026, sedangkan Januari dan Februari merupakan bulan pada tahun 2027. Jadwal tersebut bersifat indikatif dan dapat menyesuaikan dengan hasil konsultasi dosen pembimbing serta ketersediaan sumber daya komputasi pada Google Colab Pro.

\begin{table}[htbp]
\caption{Jadwal penelitian periode Maret 2026 sampai Februari 2027.}
\label{tab:jadwal}
\begin{center}
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{|c|p{3.3cm}|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{No} & \textbf{Kegiatan} & \textbf{Mar} & \textbf{Apr} & \textbf{Mei} & \textbf{Jun} & \textbf{Jul} & \textbf{Ags} & \textbf{Sep} & \textbf{Okt} & \textbf{Nov} & \textbf{Des} & \textbf{Jan} & \textbf{Feb} \\
\hline
1 & Studi literatur dan penyusunan proposal & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  &  &  \\
\hline
2 & Seminar proposal dan revisi &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  &  \\
\hline
3 & Persiapan \emph{dataset} dan lingkungan &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  \\
\hline
4 & Implementasi \emph{preprocessing} (5 teknik) &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  \\
\hline
5 & Pelatihan 10 konfigurasi ($5 \times 2$) &  &  &  &  & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  \\
\hline
6 & Evaluasi \emph{in-distribution} &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  \\
\hline
7 & Evaluasi lintas-\emph{dataset} dan Grad-CAM &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  \\
\hline
8 & Penulisan BAB IV &  &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  \\
\hline
9 & Penulisan BAB V dan revisi &  &  &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  \\
\hline
10 & Bimbingan Pembimbing I dan II & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} &  \\
\hline
11 & Seminar hasil &  &  &  &  &  &  &  &  &  &  & \cellcolor{black} &  \\
\hline
12 & Ujian skripsi dan pengumpulan final &  &  &  &  &  &  &  &  &  &  &  & \cellcolor{black} \\
\hline
\end{tabular}
\end{center}
\end{table}

## DAFTAR RUJUKAN {.unnumbered}

\refitem{Abramoff, M.~D., Lavin, P.~T., Birch, M., Shah, N., \& Folk, J.~C. (2018). Pivotal trial of an autonomous AI-based diagnostic system for detection of diabetic retinopathy in primary care offices. \emph{NPJ Digital Medicine}, 1(1), 39.}

\refitem{American Diabetes Association. (2024). Standards of Care in Diabetes: 2024. \emph{Diabetes Care}, 47(Suppl.~1), S1--S322.}

\refitem{Anupama, B.~C., Rao, S.~N., Malini, M.~B., \& Athreya, V.~V. (2025). Comparative analysis of novel preprocessing techniques and deep learning based multi-modal feature fusion for diabetic retinopathy grading. \emph{Scientific Reports}, 15, 31339.}

\refitem{Chokuwa, S., \& Khan, M.~H. (2025). Divergent domains, convergent grading: Enhancing generalization in diabetic retinopathy grading. In \emph{IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}.}

\refitem{Chopra, M., Sparrenberg, L., Berger, A., Khanna, S., Terheyden, J.~H., \& Sifa, R. (2025). From retinal pixels to patients: Evolution of deep learning research in diabetic retinopathy screening. In \emph{2025 IEEE International Conference on Big Data (IEEE BigData)}. arXiv:2511.11065.}

\refitem{Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit. \emph{Psychological Bulletin}, 70(4), 213--220.}

\refitem{Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., \& Houlsby, N. (2021). An image is worth 16$\times$16 words: Transformers for image recognition at scale. In \emph{International Conference on Learning Representations (ICLR)}.}

\refitem{Gulshan, V., Peng, L., Coram, M., Stumpe, M.~C., Wu, D., Narayanaswamy, A., \ldots\ \& Webster, D.~R. (2016). Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs. \emph{JAMA}, 316(22), 2402--2410.}

\refitem{He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep residual learning for image recognition. In \emph{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)} (pp. 770--778).}

\refitem{Li, T., Gao, Y., Wang, K., Guo, S., Liu, H., \& Kang, H. (2019). Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening. \emph{Information Sciences}, 501, 511--522.}

\refitem{Loshchilov, I., \& Hutter, F. (2019). Decoupled weight decay regularization. In \emph{International Conference on Learning Representations (ICLR)}.}

\refitem{Porwal, P., Pachade, S., Kokare, M., Deshmukh, G., Son, J., Bae, W., \ldots\ \& Meriaudeau, F. (2020). IDRiD: Diabetic retinopathy segmentation and grading challenge. \emph{Medical Image Analysis}, 59, 101561.}

\refitem{Saputra, N.~A., Helvinda, W., \& Rahman, K. (2024). Prevalence and risk factors of diabetic retinopathy in a tertiary hospital in Padang, Indonesia. \emph{Bioscientia Medicina: Journal of Biomedicine and Translational Research}, 9(1), 219--231.}

\refitem{Sasongko, M.~B., Widyaputri, F., Agni, A.~N., Wardhana, F.~S., Kotha, S., Gupta, P., Widayanti, T.~W., Haryanto, S., Widyaningrum, R., Wong, T.~Y., Kawasaki, R., \& Wang, J.~J. (2025). Incidence and progression of diabetic retinopathy and blindness in Indonesian adults with type 2 diabetes. \emph{PLoS ONE}, 20, e0322093.}

\refitem{Selvaraju, R.~R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., \& Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. In \emph{Proceedings of the IEEE International Conference on Computer Vision (ICCV)} (pp. 618--626).}

\refitem{Teo, Z.~L., Tham, Y.~C., Yu, M., Chee, M.~L., Rim, T.~H., Cheung, N., \ldots\ \& Cheng, C.~Y. (2021). Global prevalence of diabetic retinopathy and projection of burden through 2045: Systematic review and meta-analysis. \emph{Ophthalmology}, 128(11), 1580--1591.}

\refitem{Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.~N., Kaiser, L., \& Polosukhin, I. (2017). Attention is all you need. In \emph{Advances in Neural Information Processing Systems (NeurIPS)} (pp. 5998--6008).}

\refitem{Wightman, R. (2019). PyTorch Image Models (timm). GitHub repository. \texttt{https://github.com/huggingface/pytorch-image-models}}

\refitem{Wilkinson, C.~P., Ferris~III, F.~L., Klein, R.~E., Lee, P.~P., Agardh, C.~D., Davis, M., \ldots\ \& Verdaguer, J.~T. (2003). Proposed international clinical diabetic retinopathy and diabetic macular edema disease severity scales. \emph{Ophthalmology}, 110(9), 1677--1682.}

\refitem{Wong, T.~Y., \& Sabanayagam, C. (2023). The war on diabetic retinopathy: Where are we now? \emph{Asia-Pacific Journal of Ophthalmology}, 12(3), 213--221.}

\refitem{Zhou, Y., Chia, M.~A., Wagner, S.~K., Ayhan, M.~S., Williamson, D.~J., Struyven, R.~R., \ldots\ \& Keane, P.~A. (2023). A foundation model for generalizable disease detection from retinal images. \emph{Nature}, 622(7981), 156--163.}

\refitem{Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. In P.~S. Heckbert (Ed.), \emph{Graphics Gems IV} (pp. 474--485). Academic Press.}
