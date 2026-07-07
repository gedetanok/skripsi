# Pembahasan Seminar Proposal Skripsi

**Judul:** Fundus Preprocessing & Deep Learning for Diabetic Retinopathy Grading
**Penyaji:** Gede Tanok Arta Wijaya (NIM 2315101018)
**Pembimbing I:** I Made Putrama, S.T., M.Tech., Ph.D.

> Catatan: Dokumen ini berisi naskah pembahasan (talking points) per slide. Gunakan sebagai panduan saat presentasi, bukan untuk dibaca kata per kata. Setiap slide dilengkapi poin inti, narasi yang bisa diucapkan, dan transisi ke slide berikutnya. Beberapa slide juga dilengkapi antisipasi pertanyaan penguji.

---

## Slide 1 — Halaman Judul

**Tujuan slide:** Membuka presentasi dan memperkenalkan diri serta topik.

**Narasi:**
"Selamat pagi/siang Bapak/Ibu dosen penguji dan pembimbing. Perkenalkan, saya Gede Tanok Arta Wijaya, NIM 2315101018, dari Program Studi Teknik Informatika. Pada seminar proposal ini saya akan memaparkan rencana penelitian saya yang berjudul *Fundus Preprocessing & Deep Learning for Diabetic Retinopathy Grading*. Inti penelitian ini adalah analisis komparatif 5x2, yaitu lima teknik prapemrosesan citra dipadukan dengan dua arsitektur backbone, lalu diuji ketahanannya terhadap pergeseran domain antar institusi melalui evaluasi lintas dataset."

**Poin inti yang harus tersampaikan:**
- Identitas diri dan pembimbing.
- Tiga kata kunci judul: prapemrosesan fundus, deep learning, dan grading (penjenjangan keparahan) retinopati diabetik.
- Tekankan kata "komparatif" dan "lintas dataset", karena itulah dua kontribusi utama.

**Transisi:** "Berikut adalah garis besar yang akan saya sampaikan."

---

## Slide 2 — Outline

**Tujuan slide:** Memberi peta presentasi agar penguji tahu alur.

**Narasi:**
"Presentasi ini terbagi menjadi tiga bagian. Bagian pertama, Pendahuluan, membahas beban penyakit, kesenjangan kapasitas skrining, serta dua celah penelitian yang menjadi sasaran. Bagian kedua, Landasan Teori, menjelaskan penyakit retinopati diabetik dan skala ICDR, dataset yang digunakan, lima teknik prapemrosesan, dan dua backbone. Bagian ketiga, Metodologi Penelitian, memaparkan desain faktorial 5x2, alur penelitian, protokol pelatihan dan evaluasi, hingga jadwal pengerjaan."

**Poin inti:**
- Cukup sebutkan tiga pilar besar. Jangan terlalu lama di sini.
- Singgung bahwa rumusan masalah, tujuan, dan kerangka konseptual ada di dalamnya.

**Transisi:** "Mari kita mulai dari Bagian Pertama, latar belakang dan motivasi."

---

## Slide 3 — Pembuka Bagian 1: Pendahuluan & Motivasi

**Tujuan slide:** Slide pemisah bab. Cukup singkat.

**Narasi:**
"Pada bagian pendahuluan ini saya ingin menjawab dua hal: mengapa skrining retinopati diabetik otomatis itu mendesak, khususnya untuk konteks Indonesia, dan apa dua celah metodologis yang masih menghalangi akurasi riset agar bisa benar-benar dipakai di lapangan."

**Poin inti:**
- Ini slide jembatan. Cukup 1-2 kalimat lalu lanjut.

---

## Slide 4 — Latar Belakang: Beban Retinopati Diabetik

**Tujuan slide:** Menunjukkan urgensi masalah dengan data kuantitatif.

**Narasi:**
"Retinopati diabetik adalah penyebab utama kebutaan yang sebenarnya dapat dicegah pada kelompok usia produktif. Bebannya naik paling cepat justru di tempat yang kapasitas skriningnya paling lemah. Beberapa angka kunci: jumlah penderita diproyeksikan naik dari 103 juta pada tahun 2020 menjadi 161 juta pada tahun 2045. Saat ini sekitar 29 juta orang di dunia mengalami retinopati diabetik yang mengancam penglihatan. Untuk konteks Indonesia, sebuah kohort melaporkan insidensi 34,6 per 1.000 orang-tahun, dan di sebuah rumah sakit rujukan di Padang prevalensinya mencapai 55 persen pada pasien diabetes."

**Poin inti:**
- Tekankan kata "dapat dicegah" (preventable). Inilah yang membuat skrining bermakna.
- Dua angka global (161 juta, 29 juta) untuk skala dunia, dua angka Indonesia (34,6 dan 55 persen) untuk relevansi lokal.
- Sumber: Teo et al. (2021), Sasongko et al. (2025), Saputra et al. (2024), Wong & Sabanayagam (2023).

**Detail dari proposal (untuk berjaga jika ditanya):**
- Angka 34,6 berasal dari Sasongko et al. (2025), studi kohort 5 tahun atas 695 pasien diabetes tipe 2 di komunitas Yogyakarta. Studi yang sama juga mencatat kebutaan akibat DR sebesar 8,3 per 1.000 orang-tahun.
- Angka 55 persen dari Saputra et al. (2024) di rumah sakit rujukan Padang.

**Antisipasi pertanyaan:**
- *"Mengapa pakai data Padang/Yogyakarta, bukan Bali/lokasi penelitian?"* Jawab: data prevalensi dan insidensi DR berbasis populasi di Indonesia masih terbatas, dan kedua kota itu dipilih karena tersedia publikasi peer-review terbaru. Tujuannya menunjukkan beban tetap tinggi di konteks Indonesia.

**Transisi:** "Pertanyaannya, apakah kapasitas tenaga medis kita sanggup menangani beban sebesar ini?"

---

## Slide 5 — Latar Belakang: Kesenjangan Kapasitas Skrining

**Tujuan slide:** Menjelaskan mengapa otomasi menjadi satu-satunya jalan keluar.

**Narasi:**
"Jawabannya tidak. Indonesia memiliki kurang dari 2 dokter spesialis mata per 100.000 penduduk, dan itupun terkonsentrasi di kawasan urban Pulau Jawa. Padahal standar pelayanan menganjurkan pemeriksaan fundus tahunan. Dengan rasio setimpang ini, pemeriksaan manual untuk seluruh penderita diabetes secara struktural tidak mungkin dilakukan. Maka strategi yang realistis adalah interpretasi citra fundus secara otomatis di tingkat layanan primer. Intinya, hambatan utamanya bukan algoritma, melainkan kapasitas manusia. Skemanya: kamera berbiaya rendah ditempatkan di puskesmas, lalu sistem grading otomatis melakukan triase untuk menentukan siapa yang perlu dirujuk ke spesialis."

**Poin inti:**
- Kalimat kunci: "bottleneck-nya bukan algoritma, tapi kapasitas manusia."
- Otomasi diposisikan sebagai alat triase, bukan pengganti dokter. Ini penting agar tidak terkesan menggantikan peran ahli.
- Sumber: American Diabetes Association, Standards of Care in Diabetes (2024).

**Transisi:** "Lalu, otomasi seperti apa yang dibutuhkan? Bukan sekadar mendeteksi ada atau tidaknya penyakit."

---

## Slide 6 — Latar Belakang: Mengapa Grading Keparahan, Bukan Sekadar Deteksi

**Tujuan slide:** Menjelaskan sifat tugas (ordinal, 5 kelas) dan implikasi klinisnya.

**Narasi:**
"Tugas yang sebenarnya dibutuhkan bukan mendeteksi ada-tidaknya penyakit, melainkan memberi jenjang keparahan, karena setiap jenjang menentukan tindakan klinis yang berbeda. Skala ICDR membaginya menjadi lima tingkat: derajat 0 tanpa DR cukup kontrol rutin, derajat 1 NPDR ringan dipantau, derajat 2 NPDR sedang sudah mulai perlu dirujuk, derajat 3 NPDR berat dirujuk, dan derajat 4 PDR butuh penanganan segera. Ada dua sifat penting. Pertama, biaya kesalahan bersifat asimetris: kalau model menilai terlalu rendah, rujukan tertunda dan kasus terlewat; kalau terlalu tinggi, kapasitas spesialis yang langka jadi terbuang. Kedua, skala ini ordinal, sehingga salah dua tingkat lebih buruk daripada salah satu tingkat. Karena itu kita butuh metrik yang peka terhadap urutan, yaitu Quadratic Weighted Kappa atau QWK, bukan sekadar akurasi."

**Poin inti:**
- Lima kelas ICDR dan ambang rujukan di derajat 2 (moderate NPDR atau lebih buruk).
- Dua sifat: asimetri biaya kesalahan, dan sifat ordinal. Ini menjustifikasi pilihan metrik QWK nanti.
- Sumber skala: Wilkinson et al. (2003).

**Antisipasi pertanyaan:**
- *"Apa itu QWK dan mengapa bukan akurasi saja?"* Jawab: QWK memberi penalti lebih besar untuk kesalahan yang jaraknya jauh pada skala ordinal, sehingga lebih mencerminkan konsekuensi klinis dibanding akurasi yang memperlakukan semua kesalahan sama.

**Transisi:** "Secara teknis, deep learning untuk tugas ini sebenarnya sudah matang."

---

## Slide 7 — Latar Belakang: State of the Art

**Tujuan slide:** Menunjukkan bidang ini sudah matang, tapi ada celah "namun".

**Narasi:**
"Bidang ini sudah cukup matang. Pada 2016 Gulshan dan rekan menunjukkan sebuah CNN yang dilatih pada 128.175 citra mampu mendeteksi DR rujukan dengan sensitivitas dan spesifisitas di atas 90 persen. Pada 2018, IDx-DR dari Abramoff menjadi perangkat AI pertama yang mendapat izin FDA untuk penggunaan klinis otonom. Periode 2023 sampai 2025 muncul foundation model seperti RETFound yang makin mempersempit jarak antara riset dan sistem klinis. Namun, dan ini poin pentingnya, hampir semua hasil itu diperoleh pada data yang sangat rapi dan terkurasi. Hasil-hasil ini belum tentu bertahan ketika sistem dihadapkan pada kondisi lapangan yang jauh lebih heterogen, padahal di situlah deployment sesungguhnya terjadi."

**Poin inti:**
- Tiga tonggak: 2016 (setara dokter), 2018 (FDA), 2023-25 (foundation model).
- Kata sambung kunci: "...but". Bidangnya matang, tapi divalidasi pada data bersih.
- Inilah pengantar menuju dua research gap.

**Transisi:** "Kesenjangan antara data rapi dan kondisi lapangan inilah yang melahirkan dua celah penelitian saya."

---

## Slide 8 — Research Gap 1: Prapemrosesan Dipilih Secara Ad-hoc

**Tujuan slide:** Memaparkan celah pertama, fondasi untuk Faktor 1.

**Narasi:**
"Celah pertama ada pada prapemrosesan. Kualitas citra dari kamera murah sangat bervariasi: pencahayaan tidak rata, kontras rendah, dan ada derau akuisisi. Ketika lesi-lesi awal tidak terbaca, model cenderung menilai terlalu rendah, yaitu kesalahan yang justru menunda rujukan. Banyak teknik perbaikan citra tersedia seperti CLAHE, Ben Graham, Adaptive Sigmoid, LAB-ACE, dan MCIE, tetapi setiap studi biasanya hanya memilih satu teknik, tanpa perbandingan setara di bawah arsitektur dan protokol yang sama. Seperti dikutip dari Anupama dan rekan tahun 2025 di Scientific Reports, metode prapemrosesan saat ini masih terbatas pada satu fokus dan belum menjawab keragaman citra fundus secara menyeluruh, sehingga dibutuhkan eksplorasi sistematis atas kombinasi teknik. Tanpa perbandingan yang setara, pemilihan prapemrosesan masih bertumpu pada tebakan, bukan bukti."

**Poin inti:**
- Hubungkan kualitas citra buruk dengan under-grading (kesalahan paling berbahaya secara klinis).
- Masalah metodologis: tiap studi pilih satu teknik, tak ada apple-to-apple comparison.
- Kutipan Anupama et al. (2025) adalah jangkar (baseline paper) yang membenarkan penelitian ini.

**Penting untuk diketahui (dari proposal):** Anupama et al. (2025) memberi tiga rekomendasi future-work, dan penelitian ini mengambil ketiganya: (i) eksplorasi sistematis kombinasi prapemrosesan, (ii) validasi pada dataset multi-center dengan perangkat dan demografi beragam, dan (iii) integrasi alat interpretasi seperti Grad-CAM. Jadi ketika ditanya "kenapa desain seperti ini", jawabannya: ketiga pilar penelitian (perbandingan preprocessing, evaluasi lintas dataset, Grad-CAM) memetakan langsung ke tiga rekomendasi itu.

**Transisi:** "Itu celah pertama, soal prapemrosesan. Celah kedua menyangkut ketahanan model antar dataset."

---

## Slide 9 — Research Gap 2: Model Rapuh Antar Dataset

**Tujuan slide:** Memaparkan celah kedua, fondasi untuk evaluasi lintas dataset.

**Narasi:**
"Celah kedua adalah ketahanan lintas dataset. Sebuah model yang dilatih pada satu sumber data akan menurun performanya pada data di luar distribusi, bahkan ketika ontologi pelabelannya identik. Sayangnya, besarnya penurunan itu jarang diukur. Penyebabnya tiga: kamera dan protokol akuisisi berbeda antar institusi, populasi pasien berbeda, misalnya pasien Indonesia berbeda dari dataset publik tempat model dilatih, dan konsekuensinya secara klinis adalah rujukan yang terlewat sekaligus pemborosan waktu spesialis akibat rujukan palsu. Respons penelitian ini: melatih model pada IDRiD lalu mengevaluasinya secara zero-shot pada DDR. Kedua dataset memakai label yang sama tetapi berbeda perangkat, negara, dan populasi. Dengan begitu kita bisa memisahkan pengaruh pergeseran domain dari inkonsistensi anotasi, dan memperoleh estimasi realistis tentang seberapa besar penurunan yang dihadapi model ketika berpindah antar sistem kesehatan."

**Poin inti:**
- Domain shift = penurunan performa pada data luar distribusi.
- Desain IDRiD ke DDR dirancang agar label sama, sehingga perubahan murni karena domain shift, bukan beda anotasi.
- Sumber: Chokuwa & Khan (2025) serta rekomendasi future-work Anupama et al. (2025).

**Antisipasi pertanyaan:**
- *"Mengapa zero-shot, tidak fine-tune ke DDR?"* Jawab: justru tujuannya mengukur penurunan murni akibat domain shift. Kalau di-fine-tune, kita tidak lagi mengukur ketahanan, melainkan kemampuan adaptasi. Zero-shot mensimulasikan kondisi deployment ke wilayah baru tanpa anotasi ulang.

**Transisi:** "Dari dua celah ini, saya rumuskan tiga pertanyaan penelitian."

---

## Slide 10 — Rumusan Masalah: Tiga Pertanyaan Penelitian

**Tujuan slide:** Menyatakan RQ secara eksplisit. Ini slide yang paling sering ditanyakan penguji.

**Narasi:**
"Ada tiga pertanyaan penelitian. RQ1, tentang performa in-distribution: pada IDRiD, bagaimana pengaruh lima teknik prapemrosesan dan dua backbone terhadap klasifikasi keparahan DR, diukur dengan akurasi, QWK, dan macro-F1. RQ2, tentang ketahanan lintas dataset: seberapa besar penurunan tiap konfigurasi ketika diuji pada DDR tanpa adaptasi, dan konfigurasi mana yang paling tahan terhadap pergeseran domain antar institusi. RQ3, tentang trade-off deployment: bagaimana keseimbangan antara akurasi, ketahanan, dan biaya komputasi, serta konfigurasi mana yang paling cocok untuk deployment layanan primer di Indonesia."

**Poin inti:**
- RQ1 = performa, RQ2 = ketahanan, RQ3 = trade-off praktis.
- Tekankan ketiganya saling melengkapi: akurat saja tidak cukup, harus tahan dan murah.

**Transisi:** "Setiap pertanyaan ini punya tujuan yang sepadan."

---

## Slide 11 — Tujuan Penelitian

**Tujuan slide:** Memetakan tujuan ke RQ.

**Narasi:**
"Tujuan penelitian dipetakan satu-satu dengan rumusan masalah. Tujuan pertama, menganalisis pengaruh lima prapemrosesan dikali dua backbone terhadap performa pada IDRiD melalui akurasi, QWK, dan macro-F1. Tujuan kedua, mengukur secara kuantitatif penurunan performa pada DDR secara zero-shot untuk tiap konfigurasi, dan mengidentifikasi yang paling tahan domain. Tujuan ketiga, merekomendasikan konfigurasi yang paling seimbang antara akurasi, ketahanan, dan biaya komputasi untuk layanan primer Indonesia. Jadi tiap tujuan menjawab langsung RQ1 sampai RQ3."

**Poin inti:**
- Kata kerja operasional: Menganalisis, Mengukur/Quantify, Merekomendasikan.
- Tegaskan pemetaan tujuan ke RQ supaya terlihat koherensinya.

**Transisi:** "Agar penelitian tetap terukur, saya batasi ruang lingkupnya."

---

## Slide 12 — Ruang Lingkup & Batasan

**Tujuan slide:** Menunjukkan batasan yang disengaja agar eksperimen terkendali.

**Narasi:**
"Lingkup penelitian dibatasi secara sengaja agar eksperimen layak dikerjakan dan agar pengaruh prapemrosesan dan backbone terisolasi dari faktor pengganggu. Yang termasuk lingkup: grading lima kelas pada tingkat citra mengikuti skala ICDR, tanpa segmentasi atau deteksi lesi; hanya dua dataset, yaitu IDRiD untuk latih dan uji in-distribution, serta DDR untuk lintas dataset; lima teknik prapemrosesan; dan dua backbone dengan transfer learning dari ImageNet tanpa pra-pelatihan khusus domain. Yang di luar lingkup: tidak ada fine-tuning atau adaptasi pada DDR, jadi murni zero-shot; serta tidak ada studi pembaca maupun uji klinis prospektif."

**Poin inti:**
- Batasan = kekuatan, bukan kelemahan. Tujuannya isolasi variabel.
- Jelaskan dengan jujur apa yang tidak dikerjakan (zero-shot only, no clinical trial) agar penguji tidak salah harap.

**Antisipasi pertanyaan:**
- *"Mengapa hanya dua backbone dan dua dataset, tidak lebih?"* Jawab: untuk menjaga eksperimen tetap terkendali dan terjangkau secara komputasi. 5x2 sudah menghasilkan 10 konfigurasi, dan setiap penambahan faktor melipatgandakan beban pelatihan. Pipeline dirancang dapat digunakan ulang untuk menambah teknik/backbone lain di masa depan.

**Transisi:** "Lalu apa signifikansi hasilnya?"

---

## Slide 13 — Signifikansi Penelitian

**Tujuan slide:** Menjelaskan kontribusi teoretis dan praktis.

**Narasi:**
"Signifikansinya ada dua. Secara teoretis, penelitian ini memberi perbandingan empiris yang reprodusibel antara prapemrosesan dan backbone di bawah protokol terpadu, diperluas dengan evaluasi lintas dataset pada pasangan IDRiD dan DDR, sehingga langsung menjawab seruan future-work dari Anupama dan rekan tahun 2025. Secara praktis, penelitian ini menghasilkan panduan berbasis bukti untuk membangun sistem skrining DR bagi layanan primer Indonesia, sekaligus pipeline yang terdokumentasi dan dapat dipakai ulang untuk menguji teknik, backbone, atau dataset lain tanpa implementasi ulang."

**Poin inti:**
- Tiga kata kunci di chip: Reproducible, Deployment-relevant, Reusable pipeline.
- Tegaskan posisi penelitian sebagai jawaban langsung atas baseline paper (Anupama 2025).

**Transisi:** "Selanjutnya saya masuk ke landasan teori."

---

## Slide 14 — Pembuka Bagian 2: Landasan Teori

**Tujuan slide:** Slide pemisah bab.

**Narasi:**
"Pada bagian ini saya jelaskan empat fondasi: skala klinis ICDR, dataset benchmark, lima teknik prapemrosesan, dan dua arsitektur backbone, beserta alasan pemilihan masing-masing."

**Poin inti:** Singkat, lanjut.

---

## Slide 15 — Patofisiologi & Skala ICDR

**Tujuan slide:** Memberi dasar klinis singkat untuk audiens informatika.

**Narasi:**
"Secara patofisiologi, hiperglikemia kronis merusak kapiler retina dan memicu dua proses paralel. Pertama, kebocoran atau leakage: permeabilitas meningkat sehingga timbul edema dan eksudat keras. Kedua, iskemia: oklusi pembuluh memicu VEGF dan akhirnya neovaskularisasi. Lesi-lesi seperti mikroaneurisma, perdarahan, eksudat, IRMA, dan neovaskularisasi inilah yang menentukan keparahan pada skala ICDR, dan nantinya menjadi target verifikasi Grad-CAM. Skalanya: derajat 0 tanpa lesi, derajat 1 hanya mikroaneurisma, derajat 2 mulai dirujuk, derajat 3 mengikuti aturan 4-2-1, dan derajat 4 ditandai neovaskularisasi. Skala ini ordinal dan monotonik, jadi sekali lagi, salah dua tingkat lebih buruk daripada salah satu tingkat."

**Poin inti:**
- Cukup jelaskan dua mekanisme (leakage & ischaemia) dan jenis lesi.
- Hubungkan lesi ke Grad-CAM: nanti kita cek apakah model benar-benar melihat lesi.
- Aturan 4-2-1 boleh disebut singkat sebagai kriteria klinis NPDR berat (tidak perlu detail kecuali ditanya).

**Antisipasi pertanyaan:**
- *"Apa itu aturan 4-2-1?"* Jawab: kriteria untuk NPDR berat, yaitu perdarahan di 4 kuadran, atau venous beading di 2 kuadran, atau IRMA di 1 kuadran.

**Transisi:** "Untuk menguji semua ini, saya pakai dua dataset yang dipasangkan secara sengaja."

---

## Slide 16 — Dataset Benchmark: IDRiD dan DDR

**Tujuan slide:** Menjelaskan kenapa pasangan dataset ini ideal untuk mengisolasi domain shift.

**Narasi:**
"Saya memasangkan IDRiD dan DDR secara sengaja: label identik, tetapi tingkat heterogenitas berlawanan, sehingga setiap perubahan lintas dataset mencerminkan pergeseran domain, bukan perbedaan anotasi. IDRiD, dari Porwal dan rekan tahun 2020, berisi 516 citra dari satu klinik dengan satu kamera, dipakai untuk latih dan uji in-distribution. DDR, dari Li dan rekan tahun 2019, jauh lebih besar dengan 13.673 citra dari 147 rumah sakit dan beragam kamera, dipakai sebagai uji lintas dataset zero-shot, sekitar 3.759 citra setelah difilter. Keduanya memakai ontologi ICDR lima kelas dan keduanya punya paper deskriptor peer-review yang mendokumentasikan akuisisi dan split resmi."

**Poin inti:**
- IDRiD = kecil, homogen (1 klinik, 1 kamera). DDR = besar, sangat heterogen (147 RS).
- Justru kontras heterogenitas inilah yang membuat uji domain shift realistis.
- Jelaskan "filtered": DDR punya kelas ungradable yang dibuang agar konsisten dengan skala 5 kelas.

**Detail dari proposal (untuk berjaga jika ditanya):**
- IDRiD berasal dari satu klinik di Nanded, India, dengan satu jenis kamera (Kowa VX-10α). Hanya subset Disease Grading yang dipakai. Split resmi 413 latih dan 103 uji, lalu 15 persen dari data latih (62 citra) disisihkan sebagai validasi via stratified sampling, menyisakan 351 citra latih. Test 103 citra hanya diakses sekali di akhir eksperimen.
- DDR berasal dari 147 rumah sakit di 23 provinsi di China dengan beragam kamera. Hanya partisi test yang dipakai, setelah memfilter citra ungradable, menyisakan sekitar 3.759 citra. Partisi latih dan validasi DDR tidak dipakai sama sekali.

**Antisipasi pertanyaan:**
- *"Kenapa latih di IDRiD yang kecil, bukan DDR yang besar?"* Jawab: justru ini menguji skenario realistis di Indonesia, yaitu melatih pada data lokal yang terbatas lalu diterapkan luas. Selain itu, melatih pada data kecil dan homogen lalu uji pada data besar dan heterogen adalah uji ketahanan yang lebih ketat (worst-case domain shift). IDRiD kecil juga menjadi ujian nyata bagi backbone yang lapar data seperti ViT.
- *"Berapa angka 3.759 itu dari mana?"* Jawab: subset uji DDR setelah memfilter citra yang tidak dapat dinilai (ungradable) agar selaras dengan lima kelas ICDR.

**Transisi:** "Faktor pertama eksperimen ini adalah prapemrosesan."

---

## Slide 17 — Faktor 1: Lima Teknik Prapemrosesan

**Tujuan slide:** Menjelaskan kelima teknik secara ringkas dan kenapa beragam.

**Narasi:**
"Faktor pertama adalah prapemrosesan. Kelimanya menstandarkan karakteristik citra agar lesi lebih konsisten dikenali, dan dipilih untuk mencakup spektrum yang umum dipakai di literatur DR. Pertama, CLAHE, equalisasi histogram adaptif dengan batas kontras, memperkuat kontras lokal. Kedua, Ben Graham, pengurangan latar Gaussian untuk menstabilkan iluminasi global. Ketiga, Adaptive Sigmoid, peregangan kontras sigmoid dengan parameter adaptif terhadap statistik patch lokal. Keempat, LAB-ACE, menerapkan CLAHE pada kanal L di ruang warna LAB lalu direkonstruksi ke RGB. Kelima, MCIE, komposit multi-kanal yang menggabungkan kanal hijau, CLAHE, dan Ben Graham menjadi tiga kanal. Semua teknik diterapkan antara tahap cropping dan resize ke 224x224, dengan pipeline dasar yang identik untuk semua citra dan dataset."

**Poin inti:**
- Tidak perlu menghafal parameter; cukup tahu apa yang diperbaiki tiap teknik (kontras lokal vs iluminasi global vs komposit).
- Tekankan keragaman: ada yang bekerja di kontras lokal (CLAHE, LAB-ACE), iluminasi global (Ben Graham), dan komposit (MCIE). Ini penting untuk berinteraksi dengan inductive bias backbone.
- Pipeline dasar (crop, resize, urutan) identik agar perbandingan adil.

**Antisipasi pertanyaan:**
- *"Mengapa kelima teknik ini, bukan yang lain?"* Jawab: kelimanya mewakili keluarga pendekatan berbeda (histogram lokal, koreksi iluminasi, peregangan kontras, ruang warna, komposit multi-kanal) dan banyak dipakai di literatur DR, termasuk yang dirujuk Anupama et al. (2025).

**Transisi:** "Faktor kedua adalah arsitektur backbone."

---

## Slide 18 — Faktor 2: Dua Backbone, ResNet-50 dan ViT-B/16

**Tujuan slide:** Menjelaskan kedua arsitektur dan setting umumnya.

**Narasi:**
"Faktor kedua adalah backbone, dan saya pilih dua yang mewakili paradigma berlawanan. ResNet-50, jaringan residual dalam dari He dan rekan tahun 2016, mengalirkan citra melalui konvolusi 7x7, empat tahap residual, global average pooling, lalu menghasilkan 5 logit dari vektor 2048 dimensi. Bobot awalnya dari ImageNet-1k via torchvision. Kelebihannya adalah inductive bias lokal yang kuat: konvolusi membaca tekstur dan lesi halus dengan baik, dan efisien pada dataset kecil. Sebaliknya, ViT-B/16, Vision Transformer dari Dosovitskiy dan rekan tahun 2021, memecah citra menjadi patch 16x16, melewatkannya ke encoder self-attention, lalu menghasilkan 5 logit dari token CLS berdimensi 768. Bobot awalnya dari ImageNet-21k via timm. Kelebihannya self-attention global yang memodelkan konteks jarak jauh di seluruh retina dan menskala dengan pra-pelatihan besar. Keduanya saya latih penuh tanpa membekukan layer, dengan kepala klasifikasi ke 5 logit ICDR dan cross-entropy berbobot kelas."

**Poin inti:**
- ResNet = bias lokal, hemat data. ViT = attention global, butuh pra-pelatihan besar.
- Keduanya full fine-tuning (bukan frozen), agar adil.
- Bedakan sumber pretraining: ImageNet-1k vs 21k, ini relevan untuk argumen "data appetite".

**Transisi:** "Mengapa tepatnya dua arsitektur ini? Karena kontrasnya itulah yang menjadi inti eksperimen."

---

## Slide 19 — Rationale: Mengapa Dua Backbone Ini

**Tujuan slide:** Menjustifikasi pemilihan backbone sebagai variabel terkendali.

**Narasi:**
"ResNet-50 dan ViT-B/16 adalah dua paradigma dominan yang berlawanan dalam klasifikasi citra. Memasangkan keduanya mengubah pertanyaan 'backbone mana?' dari sekadar asumsi menjadi variabel terkendali. Ada empat alasan. Pertama, inductive bias-nya berlawanan: lokalitas CNN versus attention global Transformer, sehingga keduanya semestinya merespons berbeda terhadap prapemrosesan yang mengubah kontras lokal versus iluminasi global. Kedua, kebutuhan datanya berbeda: ResNet efisien pada data kecil, ViT bersandar pada pra-pelatihan skala besar, dan IDRiD yang kecil menjadi ujian nyata bagi keduanya. Ketiga, soal ketahanan masih diperdebatkan: bukti tentang mana yang lebih tahan domain shift masih bercampur, dan uji IDRiD ke DDR akan menjawabnya khusus untuk grading DR. Keempat, relevansi deployment: keduanya berbeda tajam dalam jumlah parameter dan biaya inferensi, yang langsung berkaitan dengan trade-off layanan primer di RQ3. Yang dijaga konstan: split data, augmentasi, optimiser, dan seed, sehingga perbedaan hanya bersumber dari prapemrosesan dan backbone."

**Poin inti:**
- Empat alasan: bias berlawanan, data appetite berbeda, ketahanan contested, relevansi deployment.
- Tekankan: "the contrast IS the experiment". Pemilihan backbone bukan asumsi, tapi variabel yang diuji.
- Variabel kontrol (split, augmentasi, optimiser, seed) dijaga sama.

**Transisi:** "Sekarang mari kita lihat bagaimana semua bagian ini terhubung dalam satu kerangka."

---

## Slide 20 — Kerangka Konseptual & Hipotesis

**Tujuan slide:** Menyatukan faktor, konfigurasi, outcome, dan hipotesis.

**Narasi:**
"Kerangka konseptualnya: Faktor 1, lima prapemrosesan, dikali Faktor 2, dua backbone, menghasilkan 10 konfigurasi. Kesepuluhnya dilatih pada IDRiD, lalu diukur outcome-nya berupa akurasi, QWK, ketahanan, dan biaya. Dari sini saya rumuskan tiga hipotesis. H1, pilihan prapemrosesan berpengaruh signifikan terhadap performa in-distribution. H2, konfigurasi berbeda dalam besar penurunannya pada DDR, artinya ketahanan tidak seragam. H3, konfigurasi yang paling akurat secara in-distribution belum tentu yang paling tahan. Hipotesis ketiga inilah yang paling menarik, karena memisahkan akurasi dari ketahanan."

**Poin inti:**
- Alur: 5 x 2 = 10 konfigurasi, dilatih IDRiD, diukur 4 outcome.
- Tiga hipotesis menjawab RQ1 (H1), RQ2 (H2), dan trade-off (H3).
- Tonjolkan H3: akurat belum tentu tahan, ini pesan utama penelitian.

**Penting (hipotesis di proposal bersifat berarah / directional):** Slide hanya menyebut arah umum, tetapi di proposal ada prediksi spesifik. Sampaikan ini jika ditanya "apa dugaan hasilmu":
- **H1:** teknik yang menstabilkan iluminasi global atau memfusikan representasi (Ben Graham, MCIE) diprediksi unggul, dan ViT-B/16 diprediksi minimal setara ResNet-50.
- **H2:** prapemrosesan penstandar iluminasi dan warna (Ben Graham, LAB-ACE, MCIE) dipadukan ViT-B/16 diprediksi paling tahan terhadap pergeseran domain.
- **H3:** ada trade-off antara akurasi IDRiD, ketahanan lintas dataset, dan biaya komputasi. Konfigurasi yang menyeimbangkan ketiganya diprediksi paling cocok untuk layanan primer.
- Ketiga hipotesis diuji secara empiris lewat confidence interval bootstrap.

**Transisi:** "Sekarang kita masuk ke bagaimana penelitian ini dijalankan."

---

## Slide 21 — Pembuka Bagian 3: Metodologi Penelitian

**Tujuan slide:** Slide pemisah bab.

**Narasi:**
"Bagian terakhir adalah metodologi: sebuah eksperimen faktorial kuantitatif. Saya akan jelaskan desain 5x2, alur ujung-ke-ujung, protokol pelatihan dan evaluasi, serta jadwal dua belas bulan."

**Poin inti:** Singkat, lanjut.

---

## Slide 22 — Desain Penelitian: Faktorial 5x2

**Tujuan slide:** Menunjukkan matriks 10 konfigurasi secara konkret.

**Narasi:**
"Desainnya faktorial 5x2 yang menghasilkan 10 konfigurasi, dari C01 sampai C10. Baris adalah lima prapemrosesan, kolom adalah dua backbone. Misalnya CLAHE dengan ResNet-50 adalah C01, CLAHE dengan ViT adalah C02, dan seterusnya sampai MCIE dengan ViT yaitu C10. Yang identik di kesepuluh konfigurasi: hyperparameter, augmentasi, split data, dan random seed. Dengan begitu, setiap perbedaan hasil hanya dapat ditelusuri ke prapemrosesan dan backbone, bukan faktor lain."

**Poin inti:**
- Faktorial penuh = semua kombinasi diuji, bukan sebagian.
- Penekanan pada kontrol eksperimen (semua sama kecuali dua faktor).

**Antisipasi pertanyaan:**
- *"Apakah eksperimen diulang dengan beberapa seed?"* Jawab: rencana awal satu seed tetap demi keterbandingan dan keterjangkauan komputasi, dengan confidence interval via bootstrap untuk mengukur ketidakpastian. (Bisa disebut sebagai potensi pengembangan jika sumber daya memungkinkan multi-seed.)

**Transisi:** "Bagaimana alur kerjanya secara keseluruhan?"

---

## Slide 23 — Alur Penelitian: Enam Tahap

**Tujuan slide:** Menjelaskan pipeline ujung-ke-ujung.

**Narasi:**
"Alurnya enam tahap. Tahap satu, dataset: IDRiD dan DDR dengan split resmi, plus 15 persen validasi stratified. Tahap dua, prapemrosesan: crop, resize ke 224x224, lalu satu dari lima teknik enhancement. Tahap tiga, augmentasi, hanya pada data latih, berupa flip, rotasi plus-minus 30 derajat, dan jitter kecerahan-kontras. Tahap empat, pelatihan: 10 konfigurasi dengan AdamW, cross-entropy berbobot kelas, dan early stopping berdasarkan QWK. Tahap lima, evaluasi: uji IDRiD untuk in-distribution dan uji DDR untuk lintas dataset secara zero-shot. Tahap enam, interpretasi: akurasi, QWK, macro-F1, dan Grad-CAM pada konfigurasi terbaik."

**Poin inti:**
- Augmentasi hanya di train (test tidak diaugmentasi), supaya evaluasi bersih.
- Early stopping pakai QWK, konsisten dengan metrik klinis yang ditekankan di awal.
- Grad-CAM hanya untuk konfigurasi terbaik (interpretabilitas, bukan untuk semua).

**Transisi:** "Detail pelatihan dan evaluasinya seperti berikut."

---

## Slide 24 — Protokol Pelatihan & Evaluasi

**Tujuan slide:** Memberi detail teknis yang reprodusibel.

**Narasi:**
"Untuk pelatihan: optimiser AdamW, learning rate 1e-4 dengan jadwal cosine, batch size 16, maksimum 50 epoch, dan early stopping pada QWK validasi. Karena kelas sangat tidak seimbang atau long-tailed, saya tangani di dua level: weighted sampler untuk menyeimbangkan tiap mini-batch, dan cross-entropy berbobot kelas untuk memberi penalti lebih pada kesalahan kelas langka. Saya sengaja tidak memakai mixup, label smoothing, atau focal loss, agar perbedaan hasil tetap murni dari dua faktor yang diuji. Untuk evaluasi: akurasi, QWK, macro-F1, plus confusion matrix per kelas; confidence interval 95 persen via bootstrap dengan 1.000 resampling; paired bootstrap untuk membandingkan antar konfigurasi secara statistik; dan Grad-CAM pada konfigurasi terbaik tiap backbone. Secara teknis dijalankan dengan PyTorch 2.3, torchvision, timm, OpenCV, scikit-image, di Google Colab dengan GPU T4 atau A100, sekitar 2 jam per konfigurasi atau total sekitar 20 jam."

**Poin inti:**
- Hyperparameter konkret menunjukkan kesiapan teknis.
- Penanganan imbalance dua level (sampler + weighted loss) penting karena DR datasetnya long-tailed.
- Alasan tidak pakai trik lain: menjaga atribusi ke dua faktor (clean comparison).
- Bootstrap CI dan paired bootstrap menunjukkan rigor statistik, bukan sekadar membandingkan angka mentah.

**Detail dari proposal (untuk berjaga jika ditanya):**
- Learning rate cosine turun sampai 1e-6, didahului warm-up linier 3 epoch. Weight decay 1e-4, beta1 0,9, beta2 0,999.
- Early stopping pakai patience 10 epoch; checkpoint dengan QWK validasi tertinggi yang dipakai untuk uji.
- Bobot imbalance (baik sampler maupun loss) proporsional terhadap 1/akar(n_k), dengan n_k jumlah citra kelas k. Pada kedua dataset, kelas 0 dan 2 dominan, sedangkan kelas 1, 3, dan 4 minoritas.
- Kriteria signifikansi: dua konfigurasi dianggap berbeda bermakna bila confidence interval 95 persennya tidak tumpang tindih.
- Grad-CAM dihasilkan pada konfigurasi terbaik untuk masing-masing backbone (bukan hanya satu).

**Antisipasi pertanyaan:**
- *"Mengapa tidak pakai focal loss padahal datanya imbalance?"* Jawab: secara prinsip bisa membantu, tetapi akan menambah variabel pengganggu. Tujuan eksperimen adalah mengisolasi pengaruh prapemrosesan dan backbone, jadi penanganan imbalance dijaga seminimal dan sekonsisten mungkin di semua konfigurasi.
- *"Kenapa batch size 16 saja?"* Jawab: keterbatasan memori GPU Colab, khususnya untuk ViT, dan agar konsisten di semua konfigurasi.

**Transisi:** "Terakhir, ini jadwal pengerjaannya."

---

## Slide 25 — Jadwal Penelitian: Dua Belas Bulan

**Tujuan slide:** Menunjukkan rencana waktu yang realistis.

**Narasi:**
"Penelitian direncanakan dua belas bulan, dari Maret 2026 sampai Februari 2027. Tiga bulan pertama untuk studi literatur dan penulisan proposal, lalu seminar proposal dan revisi sekitar Mei-Juni. Persiapan dataset dan lingkungan dilanjutkan implementasi lima prapemrosesan pada pertengahan tahun. Pelatihan 10 konfigurasi dan evaluasi in-distribution berlangsung sekitar Agustus-Oktober, disusul evaluasi lintas dataset dan Grad-CAM. Penulisan Bab IV dan Bab V mengikuti hasil eksperimen menjelang akhir tahun, dengan supervisi berjalan sepanjang periode. Seminar hasil sekitar Desember, dan sidang serta penyerahan akhir pada Januari-Februari 2027. Jadwal ini indikatif dan akan disesuaikan dengan konsultasi pembimbing serta ketersediaan sumber daya Colab."

**Poin inti:**
- Tunjukkan urutan logis: implementasi sebelum training, training sebelum evaluasi, evaluasi sebelum penulisan hasil.
- Supervisi membentang sepanjang periode (garis abu-abu).
- Sebut kata "indikatif" agar fleksibel jika ditanya soal kemunduran jadwal.

**Transisi:** "Demikian rencana penelitian saya."

---

## Slide 26 — Penutup / Terima Kasih

**Tujuan slide:** Menutup dan membuka sesi tanya jawab.

**Narasi:**
"Demikian pemaparan proposal saya. Sebagai ringkasan, penelitian ini melakukan perbandingan setara lima prapemrosesan dikali dua backbone untuk grading retinopati diabetik, lalu menguji ketahanannya antar institusi melalui evaluasi lintas dataset IDRiD ke DDR, dengan tujuan akhir merekomendasikan konfigurasi yang seimbang untuk layanan primer Indonesia. Saya sangat terbuka atas pertanyaan dan masukan dari Bapak dan Ibu penguji terkait desain, dataset, maupun protokol evaluasi. Terima kasih atas perhatiannya."

**Poin inti:**
- Ringkas ulang tiga kontribusi dalam satu kalimat (perbandingan setara, uji domain shift, rekomendasi deployment).
- Undang pertanyaan dengan sopan dan percaya diri.

---

## Lampiran: Daftar Pertanyaan Penguji yang Mungkin Muncul

Berikut pertanyaan lintas-slide yang sering diajukan, beserta arah jawaban singkat:

1. **"Apa kebaruan (novelty) penelitian ini?"**
   Perbandingan setara lima prapemrosesan dikali dua backbone di bawah satu protokol terpadu, ditambah kuantifikasi penurunan akibat domain shift pada pasangan IDRiD-DDR. Ini menjawab langsung future-work Anupama et al. (2025).

2. **"Mengapa hanya membandingkan, bukan mengusulkan metode baru?"**
   Karena celahnya memang pada ketiadaan perbandingan setara. Bukti komparatif yang reprodusibel adalah kontribusi yang valid dan menjadi fondasi pengembangan metode berikutnya. Penelitian ini menyediakan bukti, bukan tebakan.

3. **"Bagaimana memastikan perbandingan benar-benar adil?"**
   Semua variabel selain prapemrosesan dan backbone dijaga konstan: split data, augmentasi, optimiser, seed, dan hyperparameter. Penanganan imbalance pun dibuat seragam, tanpa trik tambahan seperti focal loss.

4. **"Apa risiko terbesar penelitian ini dan mitigasinya?"**
   Risiko: IDRiD kecil (516 citra) berpotensi overfitting, terutama untuk ViT. Mitigasi: transfer learning dari ImageNet, augmentasi, weighted sampler, early stopping pada QWK, dan pelaporan confidence interval via bootstrap.

5. **"Mengapa metrik QWK ditekankan?"**
   Karena skala ICDR ordinal, QWK memberi penalti proporsional terhadap jarak kesalahan, sehingga lebih mencerminkan konsekuensi klinis dibanding akurasi.

6. **"Bagaimana relevansinya untuk Indonesia jika datanya bukan dari Indonesia?"**
   Justru desain IDRiD ke DDR mensimulasikan pergeseran domain antar institusi dan populasi, yang merupakan tantangan utama deployment di Indonesia. Pipeline dirancang dapat dipakai ulang ketika data fundus Indonesia tersedia.

7. **"Apa output konkret yang akan dihasilkan?"**
   Tabel performa 10 konfigurasi pada IDRiD dan DDR, peringkat ketahanan terhadap domain shift, analisis trade-off akurasi-ketahanan-biaya, visualisasi Grad-CAM, dan pipeline yang terdokumentasi dan dapat digunakan ulang.

---

## Tips Penyampaian

- **Durasi:** Targetkan sekitar 15-20 menit. Slide pembuka bab (3, 14, 21) cukup 10-15 detik.
- **Slide kunci yang perlu waktu lebih:** Slide 8-9 (research gap), Slide 10 (RQ), Slide 19 (rationale backbone), dan Slide 20 (hipotesis). Di sinilah nilai akademis penelitian dinilai.
- **Jangan membaca slide.** Slide berisi poin, narasi di atas adalah penjelasannya.
- **Konsistensi istilah:** gunakan "grading" atau "penjenjangan keparahan", "prapemrosesan", "lintas dataset", "pergeseran domain" secara konsisten.
- **Saat tanya jawab:** ulangi pertanyaan penguji dengan kata-kata sendiri sebelum menjawab, untuk memastikan paham dan memberi waktu berpikir.
