<!--
  Laporan UAS (format BAB I-IV), Bahasa Indonesia.
  Sampul & metadata: laporan_build/metadata_laporan.yaml
  Build: cd laporan_build && ./build_laporan.sh
-->

## BAB I PENDAHULUAN

### 1.1 Latar Belakang

Retinopati diabetik (RD) merupakan komplikasi mikrovaskular diabetes melitus yang menjadi salah satu penyebab utama kebutaan yang dapat dicegah pada populasi usia produktif. Penyakit ini berkembang secara bertahap dan sering kali tidak menimbulkan gejala pada tahap awal, sehingga banyak penderita baru menyadarinya setelah terjadi gangguan penglihatan yang sulit dipulihkan. Oleh karena itu, penapisan dini melalui pemeriksaan citra fundus retina menjadi langkah penting dalam pencegahan kebutaan akibat diabetes.

Salah satu lesi penanda penting pada tahap RD non-proliferatif adalah *soft exudate*, yang juga dikenal sebagai *cotton wool spot*. Lesi ini terbentuk akibat iskemia pada lapisan serat saraf retina dan tampak sebagai bercak keputihan berbatas kabur pada permukaan retina. Keberadaan serta sebaran soft exudate menjadi indikator progresivitas penyakit, sehingga deteksi dan segmentasinya secara otomatis memiliki nilai klinis untuk mendukung sistem penapisan berbasis komputer, terutama di wilayah dengan keterbatasan tenaga dokter spesialis mata.

Berbeda dengan lesi terang lain seperti *hard exudate* yang berbatas tegas, soft exudate memiliki sejumlah karakteristik yang menyulitkan proses segmentasi otomatis. Pertama, ukurannya relatif kecil dan tersebar sehingga proporsi piksel lesi terhadap keseluruhan citra sangat kecil dan menimbulkan ketidakseimbangan kelas yang ekstrem. Kedua, batas tepinya kabur dengan gradien intensitas yang rendah sehingga sulit dibedakan dari latar belakang retina. Ketiga, tampilannya yang keputihan menyerupai dua struktur lain pada fundus, yaitu *optic disc* dan hard exudate, yang berpotensi memicu prediksi positif palsu apabila kedua struktur tersebut tidak ditangani terlebih dahulu. Gambar \ref{fig:repr} memperlihatkan contoh ketiga struktur tersebut pada satu citra fundus.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{gambar/representative_se.png}
\caption{Contoh citra fundus IDRiD dan anotasi lesinya. Soft exudate (merah) berbatas kabur dan tampilannya menyerupai hard exudate (kuning) maupun optic disc (biru), sehingga kedua struktur tersebut perlu dihapus sebelum segmentasi.}
\label{fig:repr}
\end{figure}

Arsitektur U-Net (Ronneberger dkk., 2015) telah menjadi tulang punggung segmentasi citra biomedis berkat struktur *encoder-decoder* dengan *skip connection* yang mempertahankan informasi spasial halus. Meski demikian, U-Net standar memperlakukan seluruh peta fitur secara seragam sehingga rentan terhadap latar belakang yang dominan ketika objek target berukuran kecil. Untuk mengatasi keterbatasan ini, diperkenalkan mekanisme atensi yang menimbang ulang pentingnya setiap fitur, salah satunya modul *concurrent spatial and channel squeeze-and-excitation* (scSE) (Roy dkk., 2018) yang menggabungkan penimbangan kanal dan spasial secara serentak dengan biaya komputasi yang ringan. U-Net yang dilengkapi modul atensi ini selanjutnya disebut Attention U-Net.

Berdasarkan permasalahan tersebut, laporan ini merancang sistem segmentasi soft exudate yang memadukan dua gagasan utama. Pertama, praproses bertingkat berbasis penghapusan struktur pengganggu, yaitu optic disc dan hard exudate dideteksi lalu dihapus dari citra sebelum proses segmentasi sehingga model tidak terdistraksi oleh objek terang non-target. Kedua, penggunaan Attention U-Net dengan encoder ResNet-34 terbobot ImageNet untuk mensegmentasi soft exudate pada citra hasil praproses. Sistem diuji pada dataset publik IDRiD untuk menilai efektivitas pendekatan yang diusulkan.

### 1.2 Rumusan Masalah

a. Bagaimana mengimplementasikan algoritma Attention U-Net dengan pipeline praproses bertingkat untuk segmentasi soft exudate pada citra fundus retina?
b. Bagaimana hasil evaluasi model Attention U-Net dalam mensegmentasi soft exudate pada citra fundus retina?

### 1.3 Tujuan

a. Mengimplementasikan algoritma Attention U-Net dengan pipeline praproses bertingkat untuk segmentasi soft exudate pada citra fundus retina.
b. Mengetahui hasil evaluasi model Attention U-Net dalam mensegmentasi soft exudate pada citra fundus retina.

### 1.4 Manfaat

a. Manfaat Teoritis. Memberikan kontribusi ilmiah mengenai penerapan Attention U-Net dan praproses berbasis penghapusan struktur pengganggu untuk segmentasi lesi retina, serta menjadi bahan referensi dan pembanding bagi penelitian selanjutnya di bidang analisis citra fundus.
b. Manfaat Praktis. Menjadi dasar pengembangan alat bantu penapisan dini retinopati diabetik yang dapat membantu tenaga medis mengidentifikasi lesi soft exudate secara otomatis, sehingga mendukung diagnosis yang lebih cepat dan konsisten.

## BAB II METODE ATTENTION U-NET

### 2.1 Konfigurasi Attention U-Net

Attention U-Net merupakan pengembangan dari arsitektur U-Net (Ronneberger dkk., 2015) yang berbentuk *encoder-decoder* dengan *skip connection*. Pada penelitian ini, encoder yang digunakan adalah ResNet-34 (He dkk., 2016) yang diinisialisasi dengan bobot praterlatih ImageNet (Deng dkk., 2009) untuk mempercepat konvergensi dan memperbaiki kemampuan ekstraksi fitur pada data medis yang terbatas. Pada jalur dekoder, setiap blok dilengkapi modul atensi *concurrent spatial and channel squeeze-and-excitation* (scSE) (Roy dkk., 2018). Secara umum, arsitektur ini terdiri atas lima bagian utama, yaitu *Input Layer*, *Encoder*, *Decoder* dengan *skip connection*, *Modul Atensi scSE*, dan *Segmentation Head*. Alur pipeline lengkap ditunjukkan pada Gambar \ref{fig:pipeline}.

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
\caption{Pipeline segmentasi soft exudate bertingkat yang diusulkan.}
\label{fig:pipeline}
\end{figure}

Arsitektur internal Attention U-Net berbentuk *encoder-decoder* simetris seperti ditunjukkan pada Gambar \ref{fig:unet}. Jalur encoder (kiri) secara bertahap memperkecil resolusi spasial sambil memperbanyak kanal fitur, sedangkan jalur dekoder (kanan) mengembalikan resolusi melalui *upsampling*. Setiap tingkat encoder terhubung ke tingkat dekoder yang setara melalui *skip connection* yang dilewatkan modul atensi scSE.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  font=\scriptsize,
  enc/.style={draw, fill=blue!12, minimum width=7mm, align=center},
  dec/.style={draw, fill=green!12, minimum width=7mm, align=center},
  bott/.style={draw, fill=orange!18, minimum width=7mm, align=center},
  scse/.style={draw, fill=red!12, rounded corners, minimum height=4.5mm, minimum width=8mm, font=\tiny, align=center},
  arr/.style={-{Stealth[length=1.6mm]}, thick},
  skip/.style={-{Stealth[length=1.6mm]}, draw=red!70, thick}]
  % encoder (kiri, menurun)
  \node[enc, minimum height=16mm] (e1) at (0,0) {$512^2$\\64};
  \node[enc, minimum height=13mm] (e2) at (1.1,-1.4) {$256^2$\\64};
  \node[enc, minimum height=10mm] (e3) at (2.2,-2.6) {$128^2$\\128};
  \node[enc, minimum height=8mm]  (e4) at (3.3,-3.6) {$64^2$\\256};
  \node[bott, minimum height=6mm] (b)  at (4.6,-4.5) {$32^2$\\512};
  % decoder (kanan, menaik)
  \node[dec, minimum height=8mm]  (d4) at (5.9,-3.6) {$64^2$\\256};
  \node[dec, minimum height=10mm] (d3) at (7.0,-2.6) {$128^2$\\128};
  \node[dec, minimum height=13mm] (d2) at (8.1,-1.4) {$256^2$\\64};
  \node[dec, minimum height=16mm] (d1) at (9.2,0) {$512^2$\\64};
  \node[draw, fill=gray!20, minimum height=16mm, minimum width=6mm, align=center] (head) at (10.4,0) {head\\$1\!\times\!1$};
  % alur encoder->bottleneck->decoder
  \draw[arr] (e1)--(e2); \draw[arr] (e2)--(e3); \draw[arr] (e3)--(e4); \draw[arr] (e4)--(b);
  \draw[arr] (b)--(d4); \draw[arr] (d4)--(d3); \draw[arr] (d3)--(d2); \draw[arr] (d2)--(d1);
  \draw[arr] (d1)--(head);
  % skip connections via scSE
  \foreach \a/\b/\y in {e1/d1/0, e2/d2/-1.4, e3/d3/-2.6, e4/d4/-3.6}{
    \node[scse] (s\a) at ($(\a)!0.5!(\b)$) {scSE};
    \draw[skip] (\a) -- (s\a); \draw[skip] (s\a) -- (\b);
  }
  \node[font=\tiny] at (0,0.95) {Encoder (ResNet-34)};
  \node[font=\tiny] at (9.2,0.95) {Decoder};
\end{tikzpicture}
\caption{Arsitektur Attention U-Net berbentuk encoder-decoder. Skip connection (panah merah) dilewatkan modul atensi scSE sebelum digabungkan ke dekoder. Angka menunjukkan resolusi spasial dan jumlah kanal.}
\label{fig:unet}
\end{figure}

#### Input Layer

Masukan model berupa citra fundus RGB berukuran $512 \times 512 \times 3$ yang telah melalui tahap praproses (penghapusan optic disc dan hard exudate serta peningkatan kontras dengan CLAHE). Ukuran citra diseragamkan agar masukan model konsisten dan efisien secara komputasi.

#### Encoder (ResNet-34)

Encoder bertugas mengekstraksi fitur dari citra masukan secara bertahap melalui serangkaian blok konvolusi residual. ResNet-34 terdiri atas lapisan konvolusi awal berukuran $7 \times 7$ diikuti empat tahap blok residual yang secara progresif memperkecil resolusi spasial sekaligus memperbanyak jumlah kanal fitur. Penggunaan koneksi residual memungkinkan pelatihan jaringan yang dalam tanpa mengalami degradasi gradien. Pada setiap tahap, peta fitur disimpan untuk diteruskan ke dekoder melalui *skip connection*.

#### Decoder dan Skip Connection

Dekoder merekonstruksi kembali resolusi spasial peta fitur secara bertahap melalui operasi *upsampling*. Pada setiap tahap, peta fitur dekoder digabungkan (*concatenate*) dengan peta fitur encoder pada resolusi yang sama melalui *skip connection*. Mekanisme ini mengembalikan informasi spasial halus yang hilang selama proses *downsampling*, sehingga model mampu menentukan batas lesi secara lebih akurat.

#### Modul Atensi scSE

Setiap blok dekoder dilengkapi modul atensi scSE yang menggabungkan dua jalur penimbangan secara serentak, sebagaimana diilustrasikan pada Gambar \ref{fig:scse}. Jalur *channel squeeze-and-excitation* (cSE) menimbang ulang pentingnya setiap kanal fitur, sedangkan jalur *spatial squeeze-and-excitation* (sSE) menimbang ulang pentingnya setiap posisi spasial. Rumus kedua jalur dijelaskan lebih lanjut pada Subbab 2.2.

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
  \node[op, right=of cse1] (cse2) {$\sigma$ skala\\kanal};
  \node[sop, below right=4mm and 12mm of u] (sse1) {Conv $1\times1$\\(sSE)};
  \node[sop, right=of sse1] (sse2) {$\sigma$ skala\\spasial};
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
\caption{Modul atensi scSE pada blok dekoder: jalur cSE menimbang kanal, jalur sSE menimbang posisi spasial, lalu keduanya dijumlahkan.}
\label{fig:scse}
\end{figure}

#### Segmentation Head

Pada ujung dekoder, peta fitur dipetakan menjadi satu kanal keluaran melalui konvolusi $1 \times 1$, lalu fungsi *sigmoid* mengubah keluaran tersebut menjadi peta probabilitas. Setiap piksel dengan probabilitas di atas ambang 0,5 ditetapkan sebagai piksel soft exudate. Ringkasan konfigurasi arsitektur disajikan pada Tabel \ref{tab:arsitektur}.

\begin{table}[H]
\centering
\caption{Konfigurasi arsitektur Attention U-Net.}
\label{tab:arsitektur}
\small
\begin{tabular}{ll}
\toprule
\textbf{Komponen} & \textbf{Spesifikasi} \\
\midrule
Ukuran masukan & $512 \times 512 \times 3$ \\
Encoder & ResNet-34 (praterlatih ImageNet) \\
Skip connection & 4 tingkat (encoder ke dekoder) \\
Modul atensi & scSE (cSE + sSE) pada tiap blok dekoder \\
Segmentation head & Konvolusi $1 \times 1$ + sigmoid \\
Jumlah kelas keluaran & 1 (soft exudate vs latar) \\
\bottomrule
\end{tabular}
\end{table}

### 2.2 Algoritma Attention U-Net

Secara umum, proses segmentasi soft exudate menggunakan Attention U-Net dapat dijelaskan melalui langkah-langkah berikut.

a. Langkah 1. Citra fundus dideteksi area optic disc-nya menggunakan Mask R-CNN, kemudian area hard exudate disegmentasi menggunakan U-Net. Kedua area tersebut dihitamkan (*blackout*) dari citra.
b. Langkah 2. Citra hasil blackout ditingkatkan kontrasnya menggunakan CLAHE pada kanal hijau, lalu diubah ukurannya menjadi $512 \times 512$ piksel.
c. Langkah 3. Citra hasil praproses dimasukkan ke encoder ResNet-34 untuk menghasilkan peta fitur bertingkat melalui blok-blok konvolusi residual.
d. Langkah 4. Dekoder melakukan *upsampling* peta fitur secara bertahap dan menggabungkannya dengan peta fitur encoder melalui *skip connection*.
e. Langkah 5. Pada setiap blok dekoder, modul atensi scSE menimbang ulang peta fitur. Jalur cSE menghitung bobot kanal melalui *global average pooling* dan dua lapis terhubung penuh:
$$
\mathbf{z}_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{U}_c(i,j), \qquad
\hat{\mathbf{U}}_{\text{cSE}} = \sigma\!\big(\mathbf{W}_2\,\delta(\mathbf{W}_1 \mathbf{z})\big) \odot \mathbf{U},
$$
dengan $\delta$ aktivasi ReLU, $\sigma$ sigmoid, serta $\odot$ perkalian per kanal.
f. Langkah 6. Jalur sSE menghitung bobot spasial melalui konvolusi $1 \times 1$:
$$
\hat{\mathbf{U}}_{\text{sSE}} = \sigma\!\big(\mathbf{W}_{sq} * \mathbf{U}\big) \odot \mathbf{U}.
$$
Keluaran scSE adalah penjumlahan kedua jalur, $\hat{\mathbf{U}}_{\text{scSE}} = \hat{\mathbf{U}}_{\text{cSE}} + \hat{\mathbf{U}}_{\text{sSE}}$.
g. Langkah 7. *Segmentation head* memetakan peta fitur akhir menjadi peta probabilitas melalui konvolusi $1 \times 1$ dan fungsi sigmoid.
h. Langkah 8. Selama pelatihan, prediksi dibandingkan dengan *ground truth* menggunakan fungsi gabungan BCE dan Dice:
$$
\mathcal{L} = \alpha\,\mathcal{L}_{\text{BCE}} + \beta\,\mathcal{L}_{\text{Dice}}, \qquad
\mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i} \hat{y}_i y_i + \varepsilon}{\sum_{i}\hat{y}_i + \sum_{i} y_i + \varepsilon},
$$
dengan $\alpha = 0{,}3$ dan $\beta = 0{,}7$.
i. Langkah 9. Parameter model diperbarui menggunakan algoritma optimasi AdamW (Loshchilov & Hutter, 2019), dan proses diulang hingga kriteria *early stopping* terpenuhi.
j. Langkah 10. Pada tahap inferensi, peta probabilitas diambang pada 0,5 untuk menghasilkan mask biner soft exudate.

### 2.3 Contoh Perhitungan Manual

Sebagai ilustrasi, dilakukan perhitungan manual modul atensi scSE pada satu peta fitur kecil. Misalkan peta fitur $\mathbf{U}$ memiliki dua kanal ($C=2$) berukuran $2 \times 2$, yaitu

$$
\mathbf{U}_1 = \begin{bmatrix} 0{,}8 & 0{,}6 \\ 0{,}4 & 0{,}2 \end{bmatrix}, \qquad
\mathbf{U}_2 = \begin{bmatrix} 0{,}1 & 0{,}3 \\ 0{,}5 & 0{,}7 \end{bmatrix}.
$$

**Langkah 1. Jalur cSE (Global Average Pooling).** Hitung rata-rata spasial tiap kanal:
$$
z_1 = \frac{0{,}8+0{,}6+0{,}4+0{,}2}{4} = 0{,}5, \qquad
z_2 = \frac{0{,}1+0{,}3+0{,}5+0{,}7}{4} = 0{,}4.
$$

**Langkah 2. Lapis terhubung penuh dan sigmoid.** Misalkan bobot reduksi $\mathbf{W}_1 = [0{,}6\ \ 0{,}4]$ dan bobot ekspansi $\mathbf{W}_2 = [0{,}7\ \ 0{,}9]^\top$. Aktivasi tersembunyi:
$$
a = \delta(0{,}6 \cdot 0{,}5 + 0{,}4 \cdot 0{,}4) = \delta(0{,}46) = 0{,}46.
$$
Bobot kanal setelah sigmoid:
$$
s_1 = \sigma(0{,}7 \cdot 0{,}46) = \sigma(0{,}322) = 0{,}580, \qquad
s_2 = \sigma(0{,}9 \cdot 0{,}46) = \sigma(0{,}414) = 0{,}602.
$$
Keluaran cSE diperoleh dengan menskala tiap kanal:
$$
\hat{\mathbf{U}}_{\text{cSE},1} = \begin{bmatrix} 0{,}464 & 0{,}348 \\ 0{,}232 & 0{,}116 \end{bmatrix}, \qquad
\hat{\mathbf{U}}_{\text{cSE},2} = \begin{bmatrix} 0{,}060 & 0{,}181 \\ 0{,}301 & 0{,}421 \end{bmatrix}.
$$

**Langkah 3. Jalur sSE (konvolusi $1 \times 1$).** Misalkan bobot konvolusi lintas kanal $\mathbf{W}_{sq} = [0{,}7\ \ 0{,}3]$. Peta atensi spasial sebelum sigmoid pada tiap posisi:
$$
q = \begin{bmatrix} 0{,}59 & 0{,}51 \\ 0{,}43 & 0{,}35 \end{bmatrix}
\;\Rightarrow\;
\sigma(q) = \begin{bmatrix} 0{,}643 & 0{,}625 \\ 0{,}606 & 0{,}587 \end{bmatrix}.
$$
Sebagai contoh, posisi $(0,0)$: $0{,}7 \cdot 0{,}8 + 0{,}3 \cdot 0{,}1 = 0{,}59$, lalu $\sigma(0{,}59) = 0{,}643$. Keluaran sSE diperoleh dengan menskala tiap posisi (berlaku untuk semua kanal):
$$
\hat{\mathbf{U}}_{\text{sSE},1} = \begin{bmatrix} 0{,}515 & 0{,}375 \\ 0{,}242 & 0{,}117 \end{bmatrix}, \qquad
\hat{\mathbf{U}}_{\text{sSE},2} = \begin{bmatrix} 0{,}064 & 0{,}187 \\ 0{,}303 & 0{,}411 \end{bmatrix}.
$$

**Langkah 4. Penjumlahan scSE.** Keluaran akhir adalah jumlah elemen kedua jalur, $\hat{\mathbf{U}}_{\text{scSE}} = \hat{\mathbf{U}}_{\text{cSE}} + \hat{\mathbf{U}}_{\text{sSE}}$:
$$
\hat{\mathbf{U}}_{\text{scSE},1} = \begin{bmatrix} 0{,}979 & 0{,}723 \\ 0{,}474 & 0{,}233 \end{bmatrix}, \qquad
\hat{\mathbf{U}}_{\text{scSE},2} = \begin{bmatrix} 0{,}124 & 0{,}368 \\ 0{,}604 & 0{,}832 \end{bmatrix}.
$$

Terlihat bahwa modul scSE menguatkan posisi dengan respons tinggi (misalnya pojok kiri atas kanal 1 yang menjadi 0,979) sekaligus menimbang kontribusi tiap kanal, sehingga fitur yang relevan terhadap lesi diperkuat dan fitur latar ditekan.

## BAB III HASIL DAN PEMBAHASAN

### 3.1 Implementasi Algoritma Attention U-Net

Alur implementasi sistem secara keseluruhan ditunjukkan pada Gambar \ref{fig:alur}, mulai dari pemuatan dataset hingga evaluasi model.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  node distance=5mm and 7mm,
  s/.style={draw, rounded corners, align=center, minimum height=9mm, minimum width=21mm, font=\scriptsize, fill=blue!6},
  arr/.style={-{Stealth[length=2mm]}, thick}]
  \node[s] (a) {Muat\\dataset};
  \node[s, right=of a] (b) {Praproses\\(OD/HE/CLAHE)};
  \node[s, right=of b] (c) {Augmentasi\\data};
  \node[s, right=of c] (d) {Bagi\\dataset};
  \node[s, below=of d] (e) {Inisialisasi\\Attention U-Net};
  \node[s, left=of e] (f) {Pelatihan\\(BCE-Dice)};
  \node[s, left=of f] (g) {Evaluasi\\model};
  \node[s, left=of g] (h) {Selesai};
  \draw[arr] (a)--(b); \draw[arr] (b)--(c); \draw[arr] (c)--(d);
  \draw[arr] (d)--(e); \draw[arr] (e)--(f); \draw[arr] (f)--(g); \draw[arr] (g)--(h);
\end{tikzpicture}
\caption{Alur implementasi sistem segmentasi soft exudate.}
\label{fig:alur}
\end{figure}

Penelitian ini menggunakan dataset publik *Indian Diabetic Retinopathy Image Dataset* (IDRiD) (Porwal dkk., 2018) pada subset segmentasi, yang terdiri atas 54 citra latih dan 27 citra uji beserta *ground truth* tingkat piksel. Anotasi soft exudate tersedia untuk 26 citra latih dan 14 citra uji, sedangkan citra tanpa lesi tetap diikutkan dengan mask kosong agar model belajar menekan prediksi positif palsu. Seluruh citra diubah ukurannya menjadi $512 \times 512$ piksel.

Tahap praproses dilakukan secara bertingkat. Pertama, area optic disc dideteksi menggunakan Mask R-CNN (He dkk., 2017) karena bentuknya tunggal dan terlokalisasi. Kedua, area hard exudate disegmentasi menggunakan U-Net terpisah karena tampilannya paling mirip dengan soft exudate. Kedua area tersebut dihitamkan dari citra. Ketiga, kontras ditingkatkan menggunakan CLAHE (Zuiderveld, 1994) pada kanal hijau, yang memberikan kontras tertinggi antara lesi dan jaringan retina. Contoh visual setiap tahapan praproses ditunjukkan pada Gambar \ref{fig:prep}.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{visualisasi_preprocessing_pipeline.png}
\caption{Tahapan praproses. Dari kiri ke kanan: citra asli, mask optic disc, mask hard exudate, citra setelah blackout, dan hasil CLAHE pada kanal hijau.}
\label{fig:prep}
\end{figure}

Pada data latih diterapkan augmentasi secara daring berupa pembalikan horizontal dan vertikal, rotasi, *shift-scale-rotate*, transformasi elastis, perubahan kecerahan dan kontras, serta penambahan derau Gaussian untuk memperbesar keragaman data dan mengurangi *overfitting*. Model dilatih menggunakan optimizer AdamW dengan laju pembelajaran awal $10^{-4}$ dan *weight decay* $10^{-4}$. Laju pembelajaran dijadwalkan dengan *ReduceLROnPlateau* (faktor 0,5; kesabaran 7 epoch berdasarkan IoU uji), disertai pemotongan norma gradien pada nilai 1,0. Pelatihan dibatasi hingga 150 epoch dengan *early stopping* berkesabaran 25 epoch. Fungsi loss yang digunakan adalah gabungan BCE dan Dice dengan bobot 0,3 dan 0,7 serta pembobotan kelas positif $w_p = 10$ untuk mengatasi ketidakseimbangan piksel. Ringkasan konfigurasi pelatihan disajikan pada Tabel \ref{tab:hyper}.

\begin{table}[H]
\centering
\caption{Konfigurasi pelatihan model.}
\label{tab:hyper}
\small
\begin{tabular}{ll}
\toprule
\textbf{Komponen} & \textbf{Nilai} \\
\midrule
Optimizer & AdamW ($\text{lr}=10^{-4}$, $\text{wd}=10^{-4}$) \\
Penjadwal lr & ReduceLROnPlateau (faktor 0,5; sabar 7) \\
Ukuran batch & 4 \\
Fungsi loss & $0{,}3\,$BCE $+\ 0{,}7\,$Dice ($w_p=10$) \\
Maksimum epoch & 150 (early stopping sabar 25) \\
Pemotongan gradien & norma maksimum 1,0 \\
Ukuran citra & $512 \times 512$ piksel \\
\bottomrule
\end{tabular}
\end{table}

### 3.2 Hasil Evaluasi Model

Perkembangan pelatihan model ditunjukkan pada Gambar \ref{fig:training}. Loss pelatihan menurun secara konsisten, sementara IoU data uji meningkat tajam pada epoch awal lalu berfluktuasi di kisaran tinggi seiring penyesuaian laju pembelajaran. IoU uji terbaik dicapai pada epoch ke-34 dari total 59 epoch sebelum *early stopping*. Spesifisitas konsisten mendekati 1,0 sejak awal, menandakan model dengan cepat belajar menekan latar belakang; tantangan utama justru pada peningkatan sensitivitas dan IoU terhadap lesi yang kecil.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{visualisasi_training_attention_unet_se.png}
\caption{Kurva pelatihan. Kiri: loss pelatihan; tengah: IoU data uji; kanan: sensitivitas dan spesifisitas data uji.}
\label{fig:training}
\end{figure}

Kinerja model pada konfigurasi terbaik dirangkum pada Tabel \ref{tab:hasil}. Model mencapai *Intersection over Union* (IoU) sebesar 0,627 dengan sensitivitas 0,805 dan spesifisitas 0,999 pada data uji. Nilai spesifisitas yang sangat tinggi menegaskan bahwa praproses penghapusan optic disc dan hard exudate berhasil menekan sumber utama positif palsu, sedangkan sensitivitas di atas 0,80 menunjukkan sebagian besar piksel soft exudate berhasil ditemukan meski ukuran dan kontrasnya rendah.

\begin{table}[H]
\centering
\caption{Hasil evaluasi segmentasi soft exudate pada data uji IDRiD.}
\label{tab:hasil}
\small
\begin{tabular}{lcc}
\toprule
\textbf{Metrik} & \textbf{Nilai} & \textbf{Persentase} \\
\midrule
IoU          & 0,627 & 62,70\% \\
Sensitivitas & 0,805 & 80,50\% \\
Spesifisitas & 0,999 & 99,90\% \\
\bottomrule
\end{tabular}
\end{table}

Sebagai pembanding, konfigurasi awal tanpa mekanisme atensi yang diuji pada eksperimen pendahuluan hanya mencapai IoU di kisaran 0,36. Penambahan modul atensi scSE pada dekoder, dipadukan dengan praproses bertingkat, meningkatkan IoU secara substansial menjadi 0,627. Hal ini menegaskan bahwa penimbangan ulang kanal dan posisi spasial secara serentak efektif mengarahkan kapasitas model ke wilayah lesi yang relevan, alih-alih tersebar pada latar belakang yang dominan. Metrik IoU yang ketat menjadikan nilai 0,627 tergolong baik untuk tugas segmentasi soft exudate yang dikenal sulit karena ukuran lesi yang kecil dan batas yang kabur.

### 3.3 Visualisasi Hasil Segmentasi

Gambar \ref{fig:pred} menampilkan perbandingan antara citra asli, citra hasil praproses, prediksi model, dan *ground truth* pada beberapa sampel uji. Secara kualitatif, model mampu menangkap lokasi dan bentuk umum bercak soft exudate dengan baik, termasuk lesi yang kecil dan tersebar. Kesalahan yang tersisa umumnya berupa batas tepi lesi yang kurang tajam, sejalan dengan sifat soft exudate yang memang berbatas kabur, serta sesekali terlewatnya bercak yang sangat samar. Pola kesalahan ini konsisten dengan profil sensitivitas dan IoU yang diperoleh.

\begin{figure}[H]
\centering
\includegraphics[width=\textwidth]{visualisasi_prediksi_se_full_pipeline.png}
\caption{Perbandingan kualitatif hasil segmentasi. Tiap baris: citra asli, citra hasil praproses, prediksi model, dan ground truth.}
\label{fig:pred}
\end{figure}

## BAB IV KESIMPULAN DAN SARAN

### 4.1 Kesimpulan

Berdasarkan hasil implementasi dan pengujian yang telah dilakukan, algoritma Attention U-Net berhasil diterapkan untuk segmentasi soft exudate pada citra fundus retina dengan dukungan pipeline praproses bertingkat. Tahapan implementasi meliputi penghapusan optic disc (Mask R-CNN) dan hard exudate (U-Net), peningkatan kontras dengan CLAHE pada kanal hijau, augmentasi data, pembangunan arsitektur Attention U-Net berbasis encoder ResNet-34 dengan modul atensi scSE, hingga pelatihan menggunakan fungsi gabungan BCE dan Dice. Modul atensi scSE memungkinkan model menimbang ulang kanal dan posisi spasial secara serentak sehingga fokus pada wilayah lesi yang relevan.

Hasil evaluasi pada dataset IDRiD menunjukkan model memperoleh IoU sebesar 0,627, sensitivitas 0,805, dan spesifisitas 0,999 pada data uji, jauh melampaui konfigurasi awal tanpa atensi yang hanya mencapai IoU sekitar 0,36. Dengan demikian, kombinasi penghapusan struktur pengganggu dan mekanisme atensi scSE terbukti efektif untuk segmentasi lesi kecil dan berbatas kabur seperti soft exudate.

### 4.2 Saran

Penelitian selanjutnya disarankan menguji pipeline ini pada beberapa dataset publik lain dengan karakteristik kamera dan populasi yang berbeda untuk menilai generalisasi model. Selain itu, ketergantungan kualitas segmentasi pada keberhasilan tahap praproses dapat dikurangi dengan mengkaji strategi *end-to-end* yang menyatukan deteksi struktur pengganggu dan segmentasi lesi dalam satu model. Eksplorasi terhadap berbagai jenis mekanisme atensi serta optimasi hiperparameter seperti ukuran batch, laju pembelajaran, dan bobot fungsi loss juga berpotensi meningkatkan kinerja segmentasi lebih lanjut.

## DAFTAR PUSTAKA {-}

\refitem{Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., \& Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. \emph{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 248--255.}

\refitem{He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep residual learning for image recognition. \emph{IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 770--778.}

\refitem{He, K., Gkioxari, G., Doll\'ar, P., \& Girshick, R. (2017). Mask R-CNN. \emph{IEEE International Conference on Computer Vision (ICCV)}, 2961--2969.}

\refitem{Loshchilov, I., \& Hutter, F. (2019). Decoupled weight decay regularization. \emph{International Conference on Learning Representations (ICLR)}.}

\refitem{Milletari, F., Navab, N., \& Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. \emph{Fourth International Conference on 3D Vision (3DV)}, 565--571.}

\refitem{Porwal, P., Pachade, S., Kamble, R., Kokare, M., Deshmukh, G., Sahasrabuddhe, V., \& Meriaudeau, F. (2018). Indian Diabetic Retinopathy Image Dataset (IDRiD): A database for diabetic retinopathy screening research. \emph{Data}, 3(3), 25.}

\refitem{Ronneberger, O., Fischer, P., \& Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. \emph{Medical Image Computing and Computer-Assisted Intervention (MICCAI)}, 234--241.}

\refitem{Roy, A. G., Navab, N., \& Wachinger, C. (2018). Concurrent spatial and channel `squeeze \& excitation' in fully convolutional networks. \emph{Medical Image Computing and Computer-Assisted Intervention (MICCAI)}, 421--429.}

\refitem{Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), \emph{Graphics Gems IV} (pp. 474--485). Academic Press.}
