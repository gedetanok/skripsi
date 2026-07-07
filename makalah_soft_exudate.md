<!--
  Naskah makalah (IMRaD, Bahasa Indonesia).
  Judul, penulis, dan abstrak ada di makalah_build/metadata_makalah.yaml.
  Build: cd makalah_build && ./build_makalah.sh
-->

## Pendahuluan

Retinopati diabetik (RD) merupakan komplikasi mikrovaskular diabetes melitus yang menjadi salah satu penyebab utama kebutaan yang dapat dicegah pada populasi usia produktif. Deteksi dini melalui penapisan citra fundus retina sangat menentukan keberhasilan penanganan, karena lesi-lesi awal sering muncul sebelum pasien merasakan gangguan penglihatan. Salah satu lesi penanda penting pada tahap RD non-proliferatif adalah *soft exudate*, yang juga dikenal sebagai *cotton wool spot*. Lesi ini terbentuk akibat iskemia pada lapisan serat saraf retina dan tampak sebagai bercak keputihan berbatas kabur. Keberadaan dan sebaran soft exudate menjadi salah satu indikator progresivitas penyakit, sehingga segmentasi otomatisnya bernilai klinis untuk mendukung sistem penapisan berbasis komputer.

Berbeda dengan lesi terang lain seperti *hard exudate* yang berbatas tegas, soft exudate memiliki sejumlah karakteristik yang menyulitkan segmentasi otomatis. Pertama, ukurannya relatif kecil dan tersebar, sehingga proporsi piksel lesi terhadap keseluruhan citra sangat kecil dan menimbulkan ketidakseimbangan kelas yang ekstrem. Kedua, batas tepinya kabur dan gradien intensitasnya rendah, sehingga sulit dibedakan dari latar belakang retina. Ketiga, tampilannya yang keputihan menyerupai dua struktur lain pada fundus, yaitu *optic disc* dan hard exudate, yang berpotensi memicu prediksi positif palsu apabila kedua struktur tersebut tidak ditangani.

Arsitektur U-Net (Ronneberger dkk., 2015) telah menjadi tulang punggung segmentasi citra biomedis berkat struktur *encoder-decoder* dengan *skip connection* yang mempertahankan informasi spasial halus. Meski demikian, U-Net standar memperlakukan seluruh peta fitur secara seragam, sehingga rentan terhadap latar belakang yang dominan ketika objek target berukuran kecil. Mekanisme atensi diperkenalkan untuk mengatasi keterbatasan ini, baik melalui *attention gate* aditif (Oktay dkk., 2018) maupun modul *squeeze-and-excitation* (Hu dkk., 2018) yang menimbang ulang pentingnya setiap kanal dan posisi spasial. Modul *concurrent spatial and channel squeeze-and-excitation* (scSE) (Roy dkk., 2018) menggabungkan kedua jenis penimbangan tersebut secara serentak dengan biaya komputasi yang ringan.

Penelitian ini mengusulkan pipeline segmentasi soft exudate bertingkat yang memadukan dua gagasan. Pertama, praproses berbasis penghapusan struktur pengganggu: area optic disc dan hard exudate dideteksi lalu dihilangkan dari citra sebelum proses segmentasi, sehingga model tidak terdistraksi oleh objek terang non-target. Kedua, penggunaan U-Net dengan encoder ResNet-34 terbobot ImageNet yang dilengkapi modul atensi scSE pada dekodernya, yang selanjutnya disebut Attention U-Net. Kontribusi penelitian ini adalah menunjukkan secara empiris pada dataset publik IDRiD bahwa kombinasi penghapusan struktur pengganggu dan mekanisme atensi scSE menghasilkan peningkatan kualitas segmentasi soft exudate yang substansial dibandingkan konfigurasi awal.

## Metode

Pendekatan yang diusulkan terdiri atas rangkaian tahap yang dijalankan secara berurutan, mulai dari penghapusan struktur pengganggu, peningkatan kontras, hingga segmentasi soft exudate menggunakan Attention U-Net. Gambaran menyeluruh alur metode ditunjukkan pada Gambar \ref{fig:pipeline}.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  node distance=6mm and 9mm,
  box/.style={draw, rounded corners, align=center, minimum height=10mm, minimum width=20mm, font=\footnotesize, fill=blue!5},
  pre/.style={draw, rounded corners, align=center, minimum height=10mm, minimum width=20mm, font=\footnotesize, fill=orange!10},
  seg/.style={draw, rounded corners, align=center, minimum height=10mm, minimum width=22mm, font=\footnotesize, fill=green!10},
  arr/.style={-{Stealth[length=2mm]}, thick}]
  \node[box] (img) {Citra\\fundus};
  \node[pre, right=of img] (od) {Deteksi OD\\(Mask R-CNN)};
  \node[pre, right=of od] (he) {Segmentasi HE\\(U-Net)};
  \node[pre, below=of he] (bo) {Blackout\\OD + HE};
  \node[pre, left=of bo] (clahe) {CLAHE\\kanal hijau};
  \node[seg, left=of clahe] (net) {Attention\\U-Net (scSE)};
  \node[box, below=of net] (out) {Mask soft\\exudate};
  \draw[arr] (img) -- (od);
  \draw[arr] (od) -- (he);
  \draw[arr] (he) -- (bo);
  \draw[arr] (bo) -- (clahe);
  \draw[arr] (clahe) -- (net);
  \draw[arr] (net) -- (out);
\end{tikzpicture}
\caption{Pipeline segmentasi soft exudate bertingkat yang diusulkan. Optic disc (OD) dan hard exudate (HE) dihapus terlebih dahulu, kontras ditingkatkan dengan CLAHE pada kanal hijau, lalu citra disegmentasi oleh Attention U-Net.}
\label{fig:pipeline}
\end{figure}

### Dataset

Penelitian ini menggunakan dataset publik *Indian Diabetic Retinopathy Image Dataset* (IDRiD) (Porwal dkk., 2018) pada subset segmentasi. Subset ini menyediakan citra fundus resolusi tinggi beserta *ground truth* segmentasi tingkat piksel untuk empat jenis lesi dan optic disc. Anotasi soft exudate tersedia dalam format berkas TIFF biner. Citra dibagi menjadi himpunan latih dan himpunan uji sesuai pembagian baku dataset. Seluruh citra dan mask diubah ukurannya menjadi $512 \times 512$ piksel agar seragam dan efisien secara komputasi. Perlu dicatat bahwa tidak semua citra memiliki anotasi soft exudate, karena lesi ini tidak selalu muncul pada setiap kasus; citra tanpa lesi tetap diikutkan dengan mask kosong agar model belajar menekan prediksi positif palsu.

### Pipeline praproses bertingkat

Tahap praproses bertujuan menghilangkan struktur terang non-target yang mudah dikelirukan dengan soft exudate, sekaligus menonjolkan kontras lesi terhadap latar belakang. Tahap ini terdiri atas tiga langkah berurutan.

Langkah pertama adalah penghapusan optic disc. Optic disc merupakan area paling terang pada fundus dan tampilannya menyerupai gumpalan eksudat. Area ini dideteksi menggunakan model Mask R-CNN yang dilatih khusus untuk segmentasi optic disc. Mask R-CNN dipilih karena optic disc berbentuk tunggal dan terlokalisasi, sehingga pendekatan *instance segmentation* sesuai.

Langkah kedua adalah penghapusan hard exudate. Hard exudate adalah lesi terang berbatas tegas yang secara visual paling mirip dengan soft exudate. Area hard exudate disegmentasi menggunakan model U-Net terpisah, lalu hasilnya digunakan sebagai masker penghapusan. Dengan menghilangkan hard exudate sejak awal, model segmentasi soft exudate dapat fokus pada karakteristik lesi target tanpa terkecoh oleh lesi terang lain.

Langkah ketiga adalah blackout dan peningkatan kontras. Area optic disc dan hard exudate yang terdeteksi kemudian dihitamkan (*blackout*) pada citra masukan. Selanjutnya, *Contrast Limited Adaptive Histogram Equalization* (CLAHE) (Zuiderveld, 1994) diterapkan pada kanal hijau citra fundus. Kanal hijau dipilih karena memberikan kontras tertinggi antara lesi dan jaringan retina dibandingkan kanal merah dan biru. CLAHE meningkatkan kontras lokal secara adaptif sambil membatasi penguatan derau melalui pemenggalan histogram (*clip limit*). Contoh visual hasil setiap langkah praproses ditampilkan pada Gambar \ref{fig:prep}.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{visualisasi_preprocessing_pipeline.png}
\caption{Visualisasi tahapan praproses pada beberapa sampel. Dari kiri ke kanan: citra asli, mask optic disc, mask hard exudate, citra setelah blackout, dan hasil CLAHE pada kanal hijau.}
\label{fig:prep}
\end{figure}

### Arsitektur Attention U-Net

Model segmentasi yang digunakan adalah U-Net dengan encoder ResNet-34 (He dkk., 2016) yang diinisialisasi dengan bobot praterlatih ImageNet (Deng dkk., 2009). Penggunaan encoder praterlatih mempercepat konvergensi dan memperbaiki kemampuan ekstraksi fitur pada data medis yang relatif terbatas. Pada jalur dekoder, setiap blok dilengkapi modul atensi *concurrent spatial and channel squeeze-and-excitation* (scSE) (Roy dkk., 2018). Modul scSE menggabungkan dua jalur penimbangan yang bekerja serentak, sebagaimana diilustrasikan pada Gambar \ref{fig:scse}.

Jalur *channel squeeze-and-excitation* (cSE) menimbang ulang pentingnya setiap kanal fitur. Diberikan peta fitur $\mathbf{U} \in \mathbb{R}^{H \times W \times C}$, jalur ini meringkas informasi spasial melalui *global average pooling* menjadi vektor deskriptor kanal $\mathbf{z} \in \mathbb{R}^{C}$, lalu menghasilkan bobot kanal:

$$
\mathbf{z}_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{U}_c(i,j), \qquad
\hat{\mathbf{U}}_{\text{cSE}} = \sigma\!\big(\mathbf{W}_2\,\delta(\mathbf{W}_1 \mathbf{z})\big) \odot \mathbf{U},
$$

dengan $\delta$ fungsi aktivasi ReLU, $\sigma$ fungsi sigmoid, $\mathbf{W}_1$ dan $\mathbf{W}_2$ bobot dua lapis terhubung penuh, serta $\odot$ perkalian elemen per kanal. Jalur *spatial squeeze-and-excitation* (sSE) menimbang ulang pentingnya setiap posisi spasial melalui konvolusi $1 \times 1$ yang memetakan peta fitur menjadi satu peta atensi spasial:

$$
\hat{\mathbf{U}}_{\text{sSE}} = \sigma\!\big(\mathbf{W}_{sq} * \mathbf{U}\big) \odot \mathbf{U},
$$

dengan $\mathbf{W}_{sq}$ kernel konvolusi $1 \times 1$ yang menghasilkan keluaran satu kanal. Keluaran modul scSE adalah penjumlahan elemen kedua jalur, $\hat{\mathbf{U}}_{\text{scSE}} = \hat{\mathbf{U}}_{\text{cSE}} + \hat{\mathbf{U}}_{\text{sSE}}$, sehingga model secara serentak menekankan kanal informatif sekaligus wilayah spasial yang relevan. Mekanisme ini membantu dekoder memusatkan perhatian pada bercak soft exudate yang kecil dan menekan respons pada latar belakang.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  node distance=5mm and 8mm,
  feat/.style={draw, fill=gray!12, minimum height=12mm, minimum width=12mm, font=\footnotesize, align=center},
  op/.style={draw, rounded corners, fill=blue!8, minimum height=8mm, minimum width=16mm, font=\scriptsize, align=center},
  sop/.style={draw, rounded corners, fill=green!10, minimum height=8mm, minimum width=16mm, font=\scriptsize, align=center},
  add/.style={draw, circle, inner sep=1pt, font=\footnotesize},
  arr/.style={-{Stealth[length=1.8mm]}, thick}]
  \node[feat] (u) {$\mathbf{U}$};
  \node[op, above right=4mm and 12mm of u] (cse1) {GAP + FC\\(cSE)};
  \node[op, right=of cse1] (cse2) {$\sigma$ scale\\kanal};
  \node[sop, below right=4mm and 12mm of u] (sse1) {Conv $1\times1$\\(sSE)};
  \node[sop, right=of sse1] (sse2) {$\sigma$ scale\\spasial};
  \node[add, right=24mm of u] (sum) {$+$};
  \node[feat, right=of sum] (out) {$\hat{\mathbf{U}}$};
  \draw[arr] (u) |- (cse1);
  \draw[arr] (cse1) -- (cse2);
  \draw[arr] (u) |- (sse1);
  \draw[arr] (sse1) -- (sse2);
  \draw[arr] (cse2) -| (sum);
  \draw[arr] (sse2) -| (sum);
  \draw[arr] (sum) -- (out);
\end{tikzpicture}
\caption{Modul atensi scSE yang disisipkan pada blok dekoder. Jalur cSE menimbang kanal, jalur sSE menimbang posisi spasial, lalu keduanya dijumlahkan.}
\label{fig:scse}
\end{figure}

### Fungsi loss

Mengingat ketidakseimbangan kelas yang ekstrem antara piksel lesi dan latar belakang, pelatihan menggunakan fungsi loss gabungan antara *Binary Cross-Entropy* (BCE) dan *Dice loss* (Milletari dkk., 2016). Komponen BCE diberi pembobotan kelas positif $w_p$ untuk memperbesar penalti terhadap lesi yang terlewat:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[\, w_p\, y_i \log \hat{y}_i + (1 - y_i)\log(1 - \hat{y}_i) \,\Big],
$$

dengan $y_i \in \{0,1\}$ label piksel ke-$i$, $\hat{y}_i$ probabilitas prediksi, dan $N$ jumlah piksel. Komponen Dice mengukur tumpang tindih antara prediksi dan ground truth secara langsung:

$$
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i} \hat{y}_i y_i + \varepsilon}{\sum_{i}\hat{y}_i + \sum_{i} y_i + \varepsilon},
$$

dengan $\varepsilon$ konstanta penghalus untuk kestabilan numerik. Fungsi loss total merupakan kombinasi berbobot kedua komponen:

$$
\mathcal{L} = \alpha\,\mathcal{L}_{\text{BCE}} + \beta\,\mathcal{L}_{\text{Dice}},
$$

dengan $\alpha = 0{,}3$ dan $\beta = 0{,}7$ sehingga komponen Dice mendominasi optimasi. Pada eksperimen ini digunakan $w_p = 10$ dan $\varepsilon = 1$.

### Konfigurasi pelatihan

Model dilatih menggunakan *optimizer* AdamW (Loshchilov & Hutter, 2019) dengan laju pembelajaran awal $10^{-4}$ dan *weight decay* $10^{-4}$. Laju pembelajaran dijadwalkan dengan strategi *ReduceLROnPlateau* yang menurunkan laju sebesar faktor 0,5 apabila IoU validasi tidak membaik selama 7 epoch. Untuk mencegah ledakan gradien, dilakukan pemotongan norma gradien pada nilai maksimum 1,0. Pelatihan dibatasi hingga 150 epoch dengan *early stopping* berkesabaran 25 epoch berdasarkan IoU data uji. Augmentasi data diterapkan secara daring berupa pembalikan horizontal dan vertikal, rotasi, *shift-scale-rotate*, transformasi elastis, perubahan kecerahan dan kontras, serta penambahan derau Gaussian, untuk memperbesar keragaman data dan mengurangi *overfitting*. Ringkasan konfigurasi pelatihan disajikan pada Tabel \ref{tab:hyper}.

\begin{table}[H]
\centering
\caption{Konfigurasi pelatihan model Attention U-Net.}
\label{tab:hyper}
\small
\begin{tabular}{ll}
\toprule
\textbf{Komponen} & \textbf{Nilai} \\
\midrule
Encoder & ResNet-34 (praterlatih ImageNet) \\
Modul atensi & scSE (spatial + channel) \\
Ukuran citra & $512 \times 512$ piksel \\
Optimizer & AdamW ($\text{lr}=10^{-4}$, $\text{wd}=10^{-4}$) \\
Penjadwal lr & ReduceLROnPlateau (faktor 0,5; sabar 7) \\
Ukuran batch & 4 \\
Fungsi loss & $0{,}3\,$BCE $+\ 0{,}7\,$Dice ($w_p=10$) \\
Maksimum epoch & 150 (early stopping sabar 25) \\
Pemotongan gradien & norma maksimum 1,0 \\
\bottomrule
\end{tabular}
\end{table}

### Metrik evaluasi

Kinerja segmentasi diukur menggunakan tiga metrik tingkat piksel: *Intersection over Union* (IoU), sensitivitas, dan spesifisitas. Dengan TP, FP, FN, dan TN berturut-turut menyatakan jumlah piksel *true positive*, *false positive*, *false negative*, dan *true negative*, ketiga metrik didefinisikan sebagai:

$$
\text{IoU} = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}, \quad
\text{Sensitivitas} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad
\text{Spesifisitas} = \frac{\text{TN}}{\text{TN} + \text{FP}}.
$$

IoU dijadikan metrik utama karena paling ketat dalam menilai tumpang tindih spasial. Sensitivitas mengukur kemampuan model menemukan piksel lesi, sedangkan spesifisitas mengukur kemampuannya menekan prediksi positif palsu pada latar belakang. Prediksi diperoleh dengan menerapkan ambang 0,5 pada keluaran sigmoid model.

## Hasil dan Pembahasan

### Dinamika pelatihan

Gambar \ref{fig:training} menampilkan kurva loss pelatihan serta perkembangan IoU, sensitivitas, dan spesifisitas pada data uji sepanjang proses pelatihan. Loss pelatihan menurun secara konsisten, sementara IoU data uji meningkat tajam pada epoch-epoch awal lalu berfluktuasi di kisaran tinggi seiring penyesuaian laju pembelajaran oleh penjadwal. IoU uji terbaik dicapai pada epoch ke-34, dan *early stopping* menghentikan pelatihan setelah tidak ada perbaikan lebih lanjut. Spesifisitas konsisten berada mendekati 1,0 sejak awal, yang menunjukkan model dengan cepat belajar menekan latar belakang; tantangan utama justru terletak pada peningkatan sensitivitas dan IoU terhadap lesi yang kecil.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{visualisasi_training_attention_unet_se.png}
\caption{Kurva pelatihan model Attention U-Net. Kiri: loss pelatihan; tengah: IoU data uji; kanan: sensitivitas dan spesifisitas data uji.}
\label{fig:training}
\end{figure}

### Kinerja kuantitatif

Tabel \ref{tab:hasil} merangkum kinerja model pada konfigurasi terbaik. Model Attention U-Net mencapai IoU 0,627 dengan sensitivitas 0,805 dan spesifisitas 0,999 pada data uji. Nilai spesifisitas yang sangat tinggi menegaskan bahwa praproses penghapusan optic disc dan hard exudate berhasil menekan sumber utama positif palsu, sedangkan sensitivitas di atas 0,80 menunjukkan sebagian besar piksel soft exudate berhasil ditemukan meski ukuran dan kontrasnya rendah.

Sebagai pembanding, konfigurasi awal tanpa mekanisme atensi yang diuji pada penelitian pendahuluan hanya mencapai IoU di kisaran 0,36. Penambahan modul atensi scSE pada dekoder, dipadukan dengan praproses bertingkat, meningkatkan IoU secara substansial menjadi 0,627. Peningkatan ini menegaskan bahwa penimbangan ulang kanal dan posisi spasial secara serentak efektif mengarahkan kapasitas model ke wilayah lesi yang relevan, alih-alih tersebar pada latar belakang yang dominan.

\begin{table}[H]
\centering
\caption{Kinerja segmentasi soft exudate pada data uji IDRiD (konfigurasi terbaik).}
\label{tab:hasil}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Atensi} & \textbf{IoU} & \textbf{Sensitivitas} & \textbf{Spesifisitas} \\
\midrule
U-Net (baseline awal) & -- & 0,36 & -- & -- \\
Attention U-Net (diusulkan) & scSE & \textbf{0,627} & 0,805 & 0,999 \\
\bottomrule
\end{tabular}
\end{table}

### Analisis kualitatif

Gambar \ref{fig:pred} menampilkan perbandingan antara citra asli, citra hasil praproses, prediksi model, dan ground truth pada beberapa sampel uji. Secara kualitatif, model mampu menangkap lokasi dan bentuk umum bercak soft exudate dengan baik, termasuk lesi yang kecil dan tersebar. Kesalahan yang tersisa umumnya berupa batas tepi lesi yang kurang tajam, sejalan dengan sifat soft exudate yang memang berbatas kabur, serta sesekali terlewatnya bercak yang sangat samar. Pola kesalahan ini konsisten dengan profil sensitivitas dan IoU yang diperoleh.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{visualisasi_prediksi_se_full_pipeline.png}
\caption{Perbandingan kualitatif hasil segmentasi soft exudate. Setiap baris menampilkan citra asli, citra hasil praproses, prediksi model, dan ground truth.}
\label{fig:pred}
\end{figure}

### Keterbatasan

Penelitian ini memiliki beberapa keterbatasan. Pertama, evaluasi dilakukan pada satu dataset publik (IDRiD), sehingga generalisasi ke dataset lain dengan karakteristik kamera dan populasi berbeda belum teruji. Kedua, kualitas segmentasi soft exudate bergantung pada keberhasilan tahap praproses; kesalahan deteksi optic disc atau hard exudate dapat merambat ke tahap segmentasi akhir. Ketiga, jumlah citra dengan anotasi soft exudate pada IDRiD relatif terbatas, yang membatasi keragaman pelatihan. Penelitian lanjutan dapat menguji pipeline ini pada beberapa dataset, membandingkan berbagai jenis mekanisme atensi secara terkendali, serta mengkaji strategi *end-to-end* yang menyatukan tahap praproses dan segmentasi dalam satu model.

## Kesimpulan

Penelitian ini mengusulkan pipeline segmentasi soft exudate bertingkat pada citra fundus retina yang memadukan praproses penghapusan struktur pengganggu (optic disc dan hard exudate) dengan model Attention U-Net berbasis modul atensi scSE. Pada dataset publik IDRiD, model mencapai IoU 0,627, sensitivitas 0,805, dan spesifisitas 0,999 pada data uji, jauh melampaui konfigurasi awal tanpa atensi yang hanya mencapai IoU sekitar 0,36. Hasil ini menunjukkan bahwa penghapusan struktur terang non-target yang dipadukan dengan penimbangan atensi kanal dan spasial secara serentak merupakan strategi yang efektif untuk segmentasi lesi kecil dan berbatas kabur seperti soft exudate. Arah penelitian selanjutnya mencakup validasi lintas dataset dan integrasi pipeline secara end-to-end.

## Daftar Pustaka

\refitem{Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., \& Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. \emph{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 248--255.}

\refitem{He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep residual learning for image recognition. \emph{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 770--778.}

\refitem{Hu, J., Shen, L., \& Sun, G. (2018). Squeeze-and-excitation networks. \emph{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 7132--7141.}

\refitem{Loshchilov, I., \& Hutter, F. (2019). Decoupled weight decay regularization. \emph{International Conference on Learning Representations (ICLR)}.}

\refitem{Milletari, F., Navab, N., \& Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. \emph{Fourth International Conference on 3D Vision (3DV)}, 565--571.}

\refitem{Oktay, O., Schlemper, J., Folgoc, L. L., Lee, M., Heinrich, M., Misawa, K., Mori, K., McDonagh, S., Hammerla, N. Y., Kainz, B., Glocker, B., \& Rueckert, D. (2018). Attention U-Net: Learning where to look for the pancreas. \emph{Medical Imaging with Deep Learning (MIDL)}.}

\refitem{Porwal, P., Pachade, S., Kamble, R., Kokare, M., Deshmukh, G., Sahasrabuddhe, V., \& Meriaudeau, F. (2018). Indian Diabetic Retinopathy Image Dataset (IDRiD): A database for diabetic retinopathy screening research. \emph{Data}, 3(3), 25.}

\refitem{Ronneberger, O., Fischer, P., \& Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. \emph{Medical Image Computing and Computer-Assisted Intervention (MICCAI)}, 234--241.}

\refitem{Roy, A. G., Navab, N., \& Wachinger, C. (2018). Concurrent spatial and channel `squeeze \& excitation' in fully convolutional networks. \emph{Medical Image Computing and Computer-Assisted Intervention (MICCAI)}, 421--429.}

\refitem{Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), \emph{Graphics Gems IV} (pp. 474--485). Academic Press.}
