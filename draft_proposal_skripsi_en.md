# THESIS PROPOSAL

**Title:**
Comparative Analysis of Fundus Image Preprocessing Techniques and Deep Learning Architectures for Diabetic Retinopathy Severity Classification across Multiple Public Datasets

---

## INTRODUCTION

### 1.1 Research Background

A progressive microvascular complication of diabetes mellitus, diabetic retinopathy (DR) stands as the foremost preventable cause of blindness in the working-age population (American Diabetes Association, 2024), and the scale of the problem is widening quickly: Teo et al. (2021) project that the 103 million adults affected in 2020 will reach roughly 161 million by 2045. Indonesia confronts this trend acutely. A DR incidence of 34.6 per 1,000 person-years (Sasongko et al., 2025) and a 55% prevalence among diabetic patients at a referral hospital (Saputra et al., 2024) occur alongside an ophthalmologist supply of fewer than two per 100,000 population that is clustered in urban Java, leaving the yearly fundus examination called for by clinical guidelines impossible to deliver by hand at the required scale. The only workable response is to automate fundus-image interpretation within primary care, where the relevant task is not simply flagging disease but assigning a severity grade on the five-level International Clinical DR (ICDR) scale (Wilkinson et al., 2003), because that grade determines whether a patient is referred. Errors are consequently costly in either direction: too low a grade postpones a referral the patient needs, whereas too high a grade consumes specialist time that is already scarce, so accuracy precisely at the grade boundaries is what counts.

Deep learning has turned automated DR grading from an aspiration into a working reality over the last ten years. Using a convolutional neural network trained on 128,175 fundus images, Gulshan et al. (2016) reported sensitivity and specificity above 90% for referable DR; Abramoff et al. (2018) subsequently secured FDA clearance for IDx-DR, the first autonomous DR system; and foundation models such as RETFound (Zhou et al., 2023) have since closed much of the remaining distance. Surveying more than fifty studies and twenty datasets, Chopra et al. (2025) confirm this maturity yet observe that the central challenge has moved away from raw capability and toward multi-center validation and clinical trust. The unresolved question, put differently, is no longer whether a model can grade DR at all, but whether the numbers obtained on carefully curated data survive the far messier conditions met in practice.

The first neglected gap lies in image quality and its treatment. Inconsistent illumination, weak contrast, and capture noise erode grading accuracy in a systematic way (Anupama et al., 2025), an effect that is worst on the inexpensive cameras typical of Indonesian primary care; because such degradation hides early lesions like microaneurysms, it pushes the model toward grades that fall below the patient's true condition. The literature offers many preprocessing methods for making lesions more visible, ranging from CLAHE, Ben Graham preprocessing, and green-channel extraction to more recent proposals such as Adaptive Sigmoid Enhancement, LAB-ACE, and Multi-channel Image Enhancement, yet they are adopted in an ad-hoc fashion, with each study committing to one method and never weighing it against the others under a shared architecture and protocol. Anupama et al. (2025) note the same problem and ask for a systematic study of preprocessing options, since without a like-for-like comparison the decision rests on guesswork instead of evidence.

The second gap concerns how far these comparisons generalise. Nearly every preprocessing comparison for DR grading, the anchor study of Anupama et al. (2025) included, is confirmed on just one dataset, so whichever technique or backbone comes out ahead there may reflect that dataset's particular cameras, population, and quality rather than a property that travels. Since fundus images differ markedly in illumination, colour, and acquisition protocol from one institution to the next, a method that prevails on one dataset will not necessarily prevail on another, and a recommendation built on a single dataset is a fragile basis for a screening system meant to face field conditions that vary widely. Anupama et al. (2025) therefore advise testing findings across several datasets that span different devices and demographics; until that is done, the external validity of any comparison of this kind stays unproven.

Taking the future-work agenda of Anupama et al. (2025) as its starting point, this study tackles both gaps together. Five preprocessing techniques are paired with two backbones (ResNet-50 and ViT-B/16) to give ten configurations, which are assessed separately on six public DR grading datasets built on the same five-grade ICDR ontology (IDRiD, DDR, APTOS 2019, Messidor-2, EyePACS, and DeepDRiD); within each dataset, every configuration is trained and tested on that dataset's own partition under one common protocol and scored by accuracy, quadratic-weighted kappa (QWK), and macro-F1. To locate the configuration that excels not on a single dataset but uniformly across all six, the per-dataset QWK values are combined through the Friedman test (Demsar, 2006) and the leading configurations are compared with the Wilcoxon signed-rank test across datasets and with per-dataset McNemar tests, while Grad-CAM checks what the leading configuration's predictions actually rest on. The contributions are threefold: (i) a reproducible, like-for-like comparison of preprocessing methods and backbones for DR grading under one unified protocol; (ii) a test of that comparison over six varied datasets to determine whether the best configuration is stable rather than tied to a particular dataset; and (iii) evidence-based guidance for building automated DR screening in resource-limited primary care.

### 1.2 Problem Identification

Based on the background described in Section 1.1, the research problems can be identified as follows.

a. The burden of diabetic retinopathy in Indonesia is high and continues to rise (an incidence of 34.6 per 1,000 person-years; Sasongko et al., 2025), while the ophthalmologist ratio is fewer than two specialists per 100,000 population and is concentrated in urban Java, so that manual annual fundus screening is structurally infeasible and demands an automated fundus-image interpretation system at the primary-care level.

b. Image quality from the low-cost cameras used in primary care is highly inconsistent (uneven illumination, weak contrast, and capture noise), and this degradation lowers automated grading accuracy and tends to make the model assign grades below the patient's true condition, the kind of error that delays a referral.

c. The way preprocessing techniques are chosen in the DR grading literature remains ad-hoc: a study usually settles on one technique without measuring it against the alternatives under a common architecture and training protocol, so like-for-like evidence to inform that choice is still missing (Anupama et al., 2025).

d. Existing preprocessing comparisons for DR grading, including the anchor study of Anupama et al. (2025), are validated on a single dataset, so it is not known whether the preprocessing and backbone identified as best on one dataset remains best on other datasets with different cameras, populations, and quality profiles, and therefore whether such a conclusion has external validity rather than being an artefact of one dataset.

e. There is as yet no evidence-based guidance on the combination of preprocessing technique and backbone architecture that performs consistently best across diverse datasets for developing automated DR screening systems for primary care with limited resources in Indonesia.

### 1.3 Research Questions

Based on the problem identification in Section 1.2, this study is formulated through the following three research questions.

a. On each of the six datasets, how do the five preprocessing techniques (CLAHE, Ben Graham preprocessing, Adaptive Sigmoid Enhancement, LAB-ACE, and Multi-channel Image Enhancement) and the two backbone architectures (ResNet-50 and Vision Transformer) affect diabetic retinopathy severity classification performance as measured by accuracy, quadratic-weighted kappa (QWK), and macro-F1?

b. When the per-dataset results are aggregated across the six datasets using the Friedman test, supported by the Wilcoxon signed-rank test and per-dataset McNemar tests on the leading configurations, which preprocessing and backbone configuration ranks consistently highest, and how stable is that ranking across datasets?

c. What is the trade-off between predictive performance and computational cost (number of parameters and inference time) for each combination of preprocessing and backbone, and which configuration is most suitable for deploying a DR screening system in primary care in Indonesia?

### 1.4 Research Limitations

So that the research stays focused and feasible to complete within the thesis timeframe, and to isolate the influence of preprocessing and backbone from other factors, the problems identified in Section 1.2 are limited as follows.

a. The classification task is limited to five-class grading of diabetic retinopathy severity on the ICDR scale at the image level, without lesion segmentation or detection.

b. The datasets used are six public DR grading datasets that share the five-class ICDR label ontology, namely IDRiD (Porwal et al., 2020), DDR (Li et al., 2019), APTOS 2019 (APTOS, 2019), Messidor-2 (Decenciere et al., 2014), EyePACS (Kaggle and EyePACS, 2015), and DeepDRiD (Liu et al., 2022). Each dataset is treated as an independent benchmark on which all configurations are trained and tested using that dataset's own partition; the study does not perform cross-dataset transfer or domain adaptation.

c. The preprocessing techniques compared are limited to five methods, namely CLAHE, Ben Graham preprocessing, Adaptive Sigmoid Enhancement, LAB-ACE, and Multi-channel Image Enhancement.

d. The backbone architectures compared are limited to ResNet-50 (CNN) and ViT-B/16 (Vision Transformer), both trained via transfer learning from ImageNet weights without domain-specific pretraining.

e. Evaluation is quantitative (accuracy, quadratic-weighted kappa, macro-F1) per dataset, aggregated across the six datasets with the Friedman test, complemented by the Wilcoxon signed-rank test across datasets and per-dataset McNemar tests (Holm-Bonferroni corrected) on the leading configurations and by Grad-CAM visualisation on the best configuration. A reader study and prospective deployment are outside the scope.

### 1.5 Research Objectives

In line with the research questions in Section 1.3, this study has the following three objectives.

a. To analyse the influence of the five preprocessing techniques (CLAHE, Ben Graham preprocessing, Adaptive Sigmoid Enhancement, LAB-ACE, and Multi-channel Image Enhancement) and the two backbone architectures (ResNet-50 and Vision Transformer) on diabetic retinopathy severity classification performance on each of the six datasets, measured through accuracy, quadratic-weighted kappa (QWK), and macro-F1.

b. To aggregate the per-dataset results across the six datasets using the Friedman test, supported by the Wilcoxon signed-rank test and per-dataset McNemar tests on the leading configurations, in order to identify the preprocessing and backbone configuration that ranks consistently highest across datasets.

c. To analyse the trade-off between predictive performance and computational cost (number of parameters and inference time) in order to recommend the configuration most suitable for deploying a diabetic retinopathy screening system in primary care in Indonesia.

### 1.6 Research Significances

On the theoretical side, the study yields a reproducible empirical comparison of preprocessing techniques against backbone architectures for DR grading under one unified protocol, reinforced by running that comparison over six public datasets that share a single label ontology yet differ in device, population, and illumination. Combining the per-dataset results through the Friedman test, supported by the Wilcoxon signed-rank test and per-dataset McNemar tests on the leading configurations, lets the study determine whether the leading configuration holds up across datasets or merely reflects one of them, which speaks directly to the future-work agenda of Anupama et al. (2025).

On the practical side, the study supplies evidence-based guidance for building automated DR screening in Indonesian primary care, a setting whose devices and populations vary widely in the field. The preprocessing-and-backbone pairing that proves consistently strongest across diverse datasets can serve as a default starting point for development. The work additionally delivers a documented experimental pipeline that can be reused to evaluate further preprocessing techniques, backbones, or datasets without rebuilding it from scratch.

## THEORETICAL FRAMEWORK

### 2.1 Diabetic Retinopathy

#### 2.1.1 Brief Pathophysiology

Diabetic retinopathy is a chronic microvascular consequence of diabetes mellitus. Sustained hyperglycaemia injures the retinal capillaries, which both become more permeable (producing oedema and lipid exudation) and occlude (producing ischaemia); that ischaemia in turn stimulates vascular endothelial growth factor (VEGF) and the growth of new vessels (Wong and Sabanayagam, 2023). The disease is split clinically into two stages. The non-proliferative stage (NPDR) presents microaneurysms, hard exudates, cotton-wool spots, intraretinal haemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA) but no new vessels, whereas the proliferative stage (PDR) is defined by neovascularisation, which can progress to vitreous haemorrhage and tractional retinal detachment. Which lesions appear, and where they sit, is what underpins the ICDR severity scale (Section 2.1.3) and is precisely what the Grad-CAM check targets (Section 2.5.3).

#### 2.1.2 Global and Indonesian Disease Burden

As detailed in Section 1.1, DR is a leading cause of preventable blindness whose global burden is projected to grow from 103 million people in 2020 to about 161 million by 2045 (Teo et al., 2021), a trajectory that Wong and Sabanayagam (2023) call the "DR pandemic". In Indonesia this burden is compounded by a severe shortage of ophthalmologists, fewer than two per 100,000 population and concentrated in urban Java, which is the structural reason automated DR screening at the primary-care level is needed (Sasongko et al., 2025; Saputra et al., 2024; American Diabetes Association, 2024).

#### 2.1.3 The International Clinical Diabetic Retinopathy (ICDR) Scale

The clinical outcome variable in this study is the International Clinical Diabetic Retinopathy (ICDR) scale, which Wilkinson et al. (2003) introduced as a simplified form of the ETDRS scale. It sorts retinopathy into five ordered grades: grade 0 for no retinopathy; grade 1 for mild NPDR with microaneurysms only; grade 2 for moderate NPDR, lying between mild and severe; grade 3 for severe NPDR, defined by the "4-2-1" rule (intraretinal haemorrhages in all four quadrants, or venous beading in two or more quadrants, or prominent IRMA, with no neovascularisation); and grade 4 for PDR, marked by neovascularisation or vitreous/pre-retinal haemorrhage.

The scale is ordinal: its five classes lie along a single increasing axis of severity, so an error spanning two grades does more harm than one spanning a single grade. That ordering calls for a metric attuned to class order, such as quadratic-weighted kappa (Section 2.5.2), rather than accuracy on its own. The cost of errors is asymmetric as well: predicting grade 1 for a true grade-3 patient denies a referral that was warranted, whereas predicting grade 1 for a true grade-0 patient ties up specialist capacity for no reason.

### 2.2 Fundus Images and Benchmark Datasets

#### 2.2.1 Colour Fundus Photography

Because it is non-invasive, inexpensive, and broadly available, colour fundus photography is the principal modality for DR screening; the non-mydriatic form, taken without dilating the pupil, is favoured for patient comfort and ease of use. How gradable an image turns out to be depends on cataract, pupil size, and how evenly it is illuminated, and some images prove ungradable, which is why current screening datasets attach a quality label so these can be filtered out.

This study uses six public DR grading datasets, each treated as an independent benchmark and described in Sections 2.2.2 to 2.2.7. They are chosen because they share the same five-class ICDR label ontology yet differ in country, acquisition device, population, and image quality; the rationale for the selection is given in Section 2.2.8, and the partition details in Section 3.2.

#### 2.2.2 Indian Diabetic Retinopathy Image Dataset (IDRiD)

Serving as the official benchmark for the IEEE ISBI 2018 challenge, the Indian Diabetic Retinopathy Image Dataset (IDRiD; Porwal et al., 2020) gathers 516 fundus images from one clinic in Nanded, India, captured with a single camera model (Kowa VX-10α). Only the Disease Grading subset is used here, where each image carries an ICDR grade from 0 to 4, and the official train and test partitions are kept unchanged so the results align with other benchmarks.

#### 2.2.3 Dataset for Diabetic Retinopathy (DDR)

The Dataset for Diabetic Retinopathy (DDR; Li et al., 2019) assembles 13,673 images drawn from 147 hospitals across 23 Chinese provinces using a range of camera types, giving it far more variety in device, quality, and demographics than IDRiD. Labels follow the ICDR scale from 0 to 4 and come with a quality label that separates gradable from ungradable images. After the ungradable images are filtered out, this study works from the official DDR train and test partitions.

#### 2.2.4 APTOS 2019 Blindness Detection Dataset

Released by the Asia Pacific Tele-Ophthalmology Society together with Aravind Eye Hospital in India, the APTOS 2019 dataset (APTOS, 2019) holds 3,662 labelled fundus images taken across rural India with several cameras under varying conditions. Grading follows the five-class ICDR scale; because the competition withholds its test labels, only the publicly labelled training set is used.

#### 2.2.5 Messidor-2

Acquired in France, Messidor-2 (Decenciere et al., 2014) comprises 1,748 macula-centred fundus images from 874 examinations. The publicly released third-party adjudicated ICDR grades, set by a panel of retina specialists, cover 1,744 gradable images and are the ones used here. Messidor-2 contributes a European population and a capture device unlike those of the Asian datasets.

#### 2.2.6 EyePACS (Kaggle Diabetic Retinopathy Detection)

As the largest public DR resource, the EyePACS dataset (Kaggle and EyePACS, 2015) gathers 88,702 fundus images captured in the United States with many camera types under widely varying conditions and graded on the five-class ICDR scale. Given that about a quarter of the images are ungradable and the collection is very large, this study draws a quality-filtered, class-stratified subset from its official partitions, as Section 3.2 details.

#### 2.2.7 DeepDRiD

The dataset of the ISBI 2020 challenge, DeepDRiD (Liu et al., 2022), supplies 2,000 regular fundus images from 500 Chinese patients, each given a five-class ICDR grade by adjudication among several ophthalmologists. This study keeps only the regular (non ultra-widefield) images and, for the dual-view cases, retains one field per eye so that every input corresponds to a single image-level grade.

#### 2.2.8 Rationale for the Six-Dataset Selection

Three considerations underpin the choice of these six datasets. First, they all use one and the same label ontology (the five-class ICDR scale; Wilkinson et al., 2003), which keeps performance comparable across datasets and lets any change in the ranking of configurations be traced to the data itself rather than to inconsistent annotation. Second, they cover dimensions that matter for deployment, namely country (India, China, France, United States), acquisition device (from a single camera at one site up to many cameras across 147 sites), population, and image-quality profile, so that a configuration which does consistently well on all six rests on solid external validity rather than the quirks of one dataset. Third, every one is public and documented (Porwal et al., 2020; Li et al., 2019; APTOS, 2019; Decenciere et al., 2014; Kaggle and EyePACS, 2015; Liu et al., 2022), meeting reproducibility needs. Above all, each dataset acts as a self-contained benchmark on which every configuration is both trained and tested, so the comparison is never contaminated by the distribution shift that transferring a model from one dataset to another would introduce.

### 2.3 Preprocessing Techniques for Fundus Images

Raw fundus images differ in quality because illumination, sensor colour response, and the optical state of the eye all vary. The purpose of preprocessing is to bring image characteristics into a common form so that clinical lesions (microaneurysms, haemorrhages, hard exudates) are picked up more consistently by the model. The five techniques assessed below span a range of approaches found in the DR grading literature, and Figure 2.1 illustrates their visual effect on a fundus image.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gambar/prep_demo.png}
\caption{The five preprocessing techniques applied to a representative colour fundus image. CLAHE and LAB-ACE amplify local contrast, Ben Graham Normalization stabilises global illumination, Adaptive Sigmoid Enhancement stretches the intensity range, and MCIE composites the green channel, CLAHE, and Ben Graham results into one three-channel input.}
\label{fig:prep}
\end{figure}

#### 2.3.1 Contrast Limited Adaptive Histogram Equalization (CLAHE)

Contrast Limited Adaptive Histogram Equalization (CLAHE; Zuiderveld, 1994) carries out histogram equalisation tile by tile. Inside a tile, a pixel value $r$ is mapped through the contrast-limited cumulative distribution $T(r) = (L-1)\sum_{j=0}^{r}\hat{p}(j)$, where $L$ counts the intensity levels and $\hat{p}$ is the tile histogram once it has been clipped at a clip limit and the excess redistributed; that clip limit is what holds noise amplification in check. For fundus images, CLAHE is run on the luminance channel (LAB) so that faint lesions such as microaneurysms, otherwise lost under uneven lighting, become visible.

#### 2.3.2 Ben Graham Normalization

Ben Graham Normalization (Graham, 2015), introduced by the winner of the Kaggle Diabetic Retinopathy Detection competition and since adopted as a routine DR grading step, removes coarse illumination by subtracting a Gaussian-blurred version of the image: $I_{\mathrm{norm}} = \alpha I + \beta G_{\sigma}(I) + \gamma$, where $G_{\sigma}$ denotes a Gaussian filter of radius $\sigma$. Fine structures such as lesions and vessels stand out more clearly, while illumination differences between cameras are damped.

#### 2.3.3 Adaptive Sigmoid Enhancement

Adaptive Sigmoid Enhancement (Anupama et al., 2025) passes intensities through a sigmoid $f(x) = 1 / (1 + \exp(-\alpha (x - \beta)))$, whose slope $\alpha$ and midpoint $\beta$ are derived adaptively from the local mean and standard deviation of intensity. The transform lifts contrast in dark areas while leaving bright-area noise untouched, so pale lesions stand out more distinctly against the retinal background.

#### 2.3.4 LAB Adaptive Contrast Enhancement (LAB-ACE)

LAB Adaptive Contrast Enhancement (LAB-ACE; Anupama et al., 2025) moves the image into LAB space and processes only the luminance channel L (CLAHE together with local normalisation), leaves the colour channels A and B as they are, and then rebuilds an RGB image. Working on L alone raises lesion contrast yet avoids the colour shifts that might otherwise mislead grading.

#### 2.3.5 Multi-channel Image Enhancement

Multi-channel Image Enhancement (MCIE; Anupama et al., 2025) combines several complementary representations into a single three-channel input, for instance the green channel (responsive to haemoglobin), CLAHE applied to the L channel, and the Ben Graham Normalization output. The backbone then draws on multiple enhancements at once rather than a single type; Anupama et al. (2025) put forward such combinations as a promising future direction.

### 2.4 Deep Learning Architectures for Image Classification

This study evaluates two deep learning backbones representing two different paradigms, namely the convolutional neural network (CNN) represented by ResNet-50 and the Vision Transformer (ViT) represented by ViT-B/16. Each architecture is described in Sections 2.4.1 and 2.4.2.

#### 2.4.1 Convolutional Neural Network and ResNet-50

A convolutional neural network (CNN) interleaves convolutions, non-linear activations ($\mathrm{ReLU}(z) = \max(0, z)$), and pooling. Sharing weights keeps the parameter count independent of image size and renders the model translation-equivariant, while greater depth assembles receptive fields that grow from edges and textures up to whole structures. The final features become logits $z \in \mathbb{R}^{K}$, which the softmax $p_k = \exp(z_k) / \sum_{j=1}^{K} \exp(z_j)$ turns into class probabilities, the whole being trained with categorical cross-entropy $\mathcal{L}_{\mathrm{CE}} = -\sum_{k=1}^{K} y_k \log p_k$.

ResNet-50 (He et al., 2016) ranks among the most heavily used CNNs in medical imaging. Its defining idea is the residual connection $y = F(x) + x$, a shortcut path that counters vanishing gradients and so makes very deep networks trainable. The network organises 50 convolutional layers, whose repeating unit is a three-layer bottleneck block carrying a residual connection, into four stages of progressively lower spatial resolution, as Figure 2.2 shows. It is taken as the CNN backbone here because it is well established, carries a moderate parameter count (~25.5 million), and recurs as a baseline throughout the DR grading literature.

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
\node[below=3pt of out, font=\scriptsize] {Bottleneck residual block: $y=\mathcal{F}(x)+x$};
\end{scope}
\draw[arr, dashed, gray] (c2.south) -- (x.north);
\end{tikzpicture}%
}
\caption{The ResNet-50 architecture: a $7\times7$ convolution and max pooling followed by four residual stages (conv2\_x to conv5\_x) built from bottleneck blocks, then global average pooling and a fully connected layer. The inset shows the bottleneck residual block with its identity shortcut, $y=\mathcal{F}(x)+x$. In this study the classification head outputs five ICDR classes. Source: adapted from He et al. (2016).}
\label{fig:resnet}
\end{figure}

#### 2.4.2 Vision Transformer (ViT)

The Vision Transformer (ViT; Dosovitskiy et al., 2021) carries the Transformer architecture (Vaswani et al., 2017) over to image recognition. It cuts the image into fixed-size patches ($16 \times 16$ pixels for ViT-B/16), flattens each patch into a token, attaches positional embeddings, and feeds the tokens through a stack of Transformer encoder blocks. At its heart is self-attention, which lets every token gauge how relevant all the others are by way of query $Q$, key $K$, and value $V$ projections: $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^{\top}/\sqrt{d_k}) V$. Modelling such long-range dependencies suits DR grading, where lesions may be spread across different retinal quadrants. Figure 2.3 lays out the ViT-B/16 architecture along with the Transformer encoder block and its multi-head self-attention. Anupama et al. (2025) name ViT outright as a direction worth pursuing for DR grading.

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{gambar/vit_paper.png}
\caption{The Vision Transformer architecture. An image is split into fixed-size patches that, together with position embeddings and a learnable [class] token, are processed by a Transformer encoder; the right panel shows the encoder block with multi-head self-attention. In this study the input is a fundus image and the classification head outputs five ICDR classes. Source: Dosovitskiy et al. (2021); encoder block after Vaswani et al. (2017).}
\label{fig:vit}
\end{figure}

#### 2.4.3 Transfer Learning

Public DR datasets are small (IDRiD offers only a few hundred training images), so starting either backbone from random weights would invite overfitting. Transfer learning instead begins from ImageNet-pretrained weights that already capture general visual features and fine-tunes on the target data after swapping the classifier head for five classes, which lowers the data demand and accelerates convergence enough to make training on IDRiD practical.

### 2.5 Evaluation Metrics for DR Grading Classification

#### 2.5.1 Accuracy, Precision, Recall, and Macro-F1

Accuracy, the share of correct predictions $\mathrm{Accuracy} = (1/N) \sum_{i=1}^{N} \mathbb{1}[\hat{y}_{i} = y_{i}]$, is misleading on imbalanced data where grade 0 predominates, as in IDRiD and DDR. The study therefore also reports per-class precision and recall alongside macro-F1, the unweighted mean over all classes of $F1 = 2\,(\mathrm{precision} \cdot \mathrm{recall}) / (\mathrm{precision} + \mathrm{recall})$. By weighting every class equally, macro-F1 stays responsive to how well the minority classes are handled.

#### 2.5.2 Quadratic-weighted Kappa (QWK)

Since ICDR is ordinal, an error of two grades is worse than an error of one, a distinction that accuracy and F1 ignore. Quadratic-weighted kappa (Cohen, 1968) applies a penalty that grows with the square of the gap between classes,

$$
\kappa_{w} \;=\; 1 - \frac{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} O_{ij}}{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} E_{ij}},
\qquad w_{ij} = \frac{(i - j)^{2}}{(K - 1)^{2}},
$$

in which $O$ is the observed confusion matrix and $E$ is the confusion matrix expected were predictions and labels independent. Here $\kappa_w = 1$ signals perfect agreement and $\kappa_w = 0$ corresponds to chance-level guessing. Established as the standard score in DR grading challenges (Kaggle Diabetic Retinopathy Detection, APTOS 2019), QWK is the primary metric of this study.

#### 2.5.3 Gradient-weighted Class Activation Mapping (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) yields a heatmap of the regions that most sway the classification, obtained by weighting the final convolutional layer's feature maps with the gradient of the target class score. For DR grading, it helps confirm whether a prediction is grounded in clinically meaningful lesions (such as microaneurysms or haemorrhages) rather than in artefacts like the lens edge. The present study treats Grad-CAM as a supporting interpretation tool applied to the best configuration rather than as the main quantitative concern, in keeping with the recommendation of Anupama et al. (2025).

### 2.6 Related Work

As Section 1.1 noted, deep-learning DR grading attained clinical and regulatory maturity through landmark systems (Gulshan et al., 2016; Abramoff et al., 2018); the subsequent release of public benchmarks such as IDRiD (Porwal et al., 2020) and DDR (Li et al., 2019) then turned attention to openly comparing backbones (ResNet, DenseNet, Inception, EfficientNet) and, more recently, foundation models such as RETFound (Zhou et al., 2023). The present study belongs to this open-benchmark tradition but concentrates on two questions that remain underexplored: how preprocessing and backbone interact, and whether the comparison that results carries over across datasets.

Chokuwa and Khan (2025) demonstrated that DR grading models lose substantial performance when one trained on a given dataset is tested on another whose acquisition settings differ, the cause being domain shift (camera, illumination, demographics) rather than inconsistent annotation. That result is why this study declines to move a single model between datasets and instead treats each dataset as a self-contained benchmark, asking whether the comparative verdict, that is, which preprocessing and backbone come out best, stays the same across datasets.

This study's anchor paper is Anupama et al. (2025) in Scientific Reports, which assesses several backbones for DR grading on one dataset and, in its future-work section, points to three directions: (i) a systematic study of preprocessing combinations; (ii) validation on multi-center datasets covering diverse devices and demographics; and (iii) the addition of interpretation tools such as Grad-CAM. The present work builds on all three, comparing five preprocessing techniques (Section 2.3) across two backbones from different paradigms (Section 2.4), evaluating them independently over six public datasets (Section 2.2), and applying Grad-CAM to the best configuration (Section 2.5.3).

### 2.7 Conceptual Framework

Raw fundus images vary in illumination, contrast, and colour, and the earliest DR lesions are small and low in contrast (Sections 2.1 and 2.2). Since the five preprocessing techniques (Section 2.3) and the two backbones (Section 2.4) operate through different mechanisms and inductive biases, performance is viewed as a function of the preprocessing $\times$ backbone interaction, which the $5 \times 2$ factorial design (Section 3.1) is constructed to isolate. To check whether the interaction extends past a single dataset, the ten configurations are trained and evaluated independently on six datasets that differ in camera, population, and illumination yet share the ICDR ontology, so that a configuration ranking first consistently on all six can claim external validity. Because the ICDR scale is ordinal and the classes are imbalanced, the comparison leans on QWK and macro-F1 instead of accuracy alone (Section 2.5) and pools the per-dataset QWK through the Friedman test, with the leading configurations then compared by the Wilcoxon signed-rank test and per-dataset McNemar tests (Section 3.7). These considerations lead to the hypotheses set out in Section 2.8.

### 2.8 Research Hypotheses

Following the research questions, this study is guided by a single statistical hypothesis, tested as described in Section 3.7. For any pair of configurations, denoted Configuration A and Configuration B, the hypotheses are defined as follows.

- $H_{0(A,B)}$: the median difference between the QWK scores of Configuration A and Configuration B across the six datasets is zero; that is, neither configuration tends to outperform the other, and any observed difference is due to random chance.
- $H_{1(A,B)}$: the median difference between the QWK scores of Configuration A and Configuration B across the six datasets is not zero; that is, one configuration systematically performs better than the other across the datasets.

The trade-off between predictive performance and computational cost (RQ3) is examined descriptively through direct measurement and is therefore not stated as a statistical hypothesis.

## RESEARCH METHODOLOGY

### 3.1 Research Design

The study adopts a quantitative factorial experimental design. Its two independent factors are the fundus-image preprocessing technique at five levels (CLAHE, Ben Graham Normalization, Adaptive Sigmoid Enhancement, LAB-ACE, and MCIE) and the backbone architecture at two levels (ResNet-50 and ViT-B/16), which combine into ten configurations ($5 \times 2$); their operational definitions appear in Section 3.3. Each of the six datasets (IDRiD, DDR, APTOS 2019, Messidor-2, EyePACS, and DeepDRiD) serves as an independent benchmark: on every dataset the ten configurations are trained and tested on that dataset's own partition under one identical protocol and scored by accuracy, quadratic-weighted kappa (QWK), macro-F1, and per-class confusion matrices. The outcome is a $10 \times 6$ matrix of per-dataset QWK scores.

To determine which configuration is best not on a single dataset but consistently across all six, the per-dataset QWK scores are aggregated with the Friedman test (Demsar, 2006) as an omnibus test, and the two leading configurations are then compared across datasets with the Wilcoxon signed-rank test and within each dataset with Holm-Bonferroni-corrected McNemar tests on the paired predictions (Section 3.7). All hyperparameters, the augmentation pipeline, the data split per dataset, and the random seed are held identical across configurations, so observed differences within a dataset are attributable to the preprocessing technique and the backbone choice alone. Model weights, seeds, scripts, and metric logs are stored in a public Git repository for reproducibility. Figure 3.1 presents the six-stage flow, from per-dataset splitting through preprocessing, training, and evaluation to the statistical aggregation and Grad-CAM on the best configuration.

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
% ---- Row 1: per-dataset factorial experiment ----
\node[db] (data) at (0,0) {Each of\\6 datasets};
\node[sbox] (basic) at (2.7,0) {Basic pipeline:\\crop, resize\\$224^2$, augment};
\node[pbox] (p1) at (5.9, 3.0) {CLAHE};
\node[pbox] (p2) at (5.9, 1.5) {Ben Graham};
\node[pbox] (p3) at (5.9, 0.0) {Adaptive Sigmoid};
\node[pbox] (p4) at (5.9,-1.5) {LAB-ACE};
\node[pbox] (p5) at (5.9,-3.0) {MCIE};
\node[bbox] (b1) at (9.4, 1.1) {ResNet-50};
\node[bbox] (b2) at (9.4,-1.1) {ViT-B/16};
\node[sbox] (conf) at (12.4,0) {$5 \times 2 = 10$\\configurations};
\node[sbox] (perds) at (15.3,0) {Train \& test\\on each dataset\\$\to$ QWK};
\node[font=\scriptsize\itshape, text=gray] at (5.9,3.95) {Factor 1: preprocessing};
\node[font=\scriptsize\itshape, text=gray] at (9.4,2.15) {Factor 2: backbone};
\draw[arr] (data) -- (basic);
\foreach \p in {p1,p2,p3,p4,p5}{\draw[fan] (basic.east) -- (\p.west);}
\foreach \p in {p1,p2,p3,p4,p5}{\draw[fan] (\p.east) -- (b1.west); \draw[fan] (\p.east) -- (b2.west);}
\draw[arr] (b1.east) -- (conf.north west);
\draw[arr] (b2.east) -- (conf.south west);
\draw[arr] (conf) -- (perds);
% ---- Row 2: aggregation across the six datasets ----
\node[sbox] (matrix) at (15.3,-5.3) {$10 \times 6$\\QWK matrix};
\node[sbox] (fried) at (11.1,-5.3) {Friedman test\\(omnibus)};
\node[obox] (cd) at (7.0,-5.3) {Wilcoxon $+$ McNemar\\on top-2 configs};
\node[obox] (best) at (2.9,-5.3) {Best configuration\\$+$ Grad-CAM};
\node[font=\scriptsize\itshape, text=gray] at (15.3,-3.9) {repeat for all 6 datasets};
\draw[arr] (perds.south) -- (matrix.north);
\draw[arr] (matrix) -- (fried);
\draw[arr] (fried) -- (cd);
\draw[arr] (cd) -- (best);
\end{tikzpicture}%
}
\caption{Research design. For each of the six datasets, an image passes through the basic pipeline and then branches into the five preprocessing techniques (Factor 1); each branch is paired with both backbones (Factor 2), forming ten configurations that are trained and tested on that dataset to produce its QWK. Repeating this for all six datasets yields a $10 \times 6$ QWK matrix, which is aggregated with the Friedman test and then examined with the Wilcoxon signed-rank and McNemar tests on the two leading configurations to identify the best configuration, finally verified with Grad-CAM.}
\label{fig:pipeline}
\end{figure}

### 3.2 Dataset

#### 3.2.1 Per-Dataset Train, Validation, and Test Partitions

Each of the six datasets is split into train, validation, and test partitions independently, and all ten configurations are trained and evaluated within each dataset using these partitions. The validation partition is used for early stopping and checkpoint selection; the test partition is accessed only once per configuration, at the end of training, to produce the reported metrics. For datasets that provide official partitions, those are adopted unchanged; for the others, a stratified split under a fixed seed is used.

For **IDRiD**, the official partition (413 training, 103 test; Porwal et al., 2020) is used, with 15% of the training images (62 images) set aside as a stratified validation partition under a fixed seed, leaving 351 training images. For **DDR** (Li et al., 2019), the official train, validation, and test partitions are used after ungradable images are filtered out. For **DeepDRiD** (Liu et al., 2022) and **EyePACS** (Kaggle and EyePACS, 2015), the official partitions are likewise adopted; for EyePACS, ungradable images are removed and a class-stratified subset is drawn to keep the training cost manageable while preserving the class distribution. For **APTOS 2019** (APTOS, 2019) and **Messidor-2** (Decenciere et al., 2014), which do not provide official train and test partitions, a stratified 70/15/15 train/validation/test split under a fixed seed is used. All splits are released in the public repository so that they can be reproduced exactly.

#### 3.2.2 Class Distribution and Handling of Imbalance

All six datasets are imbalanced (long-tailed): grade 0 dominates while the higher grades, especially grades 3 and 4, are minorities. Within each dataset the imbalance is handled identically at two levels, namely (i) a weighted random sampler with weights $\propto 1/\sqrt{n_k}$ ($n_k$ is the number of training images of class $k$ in that dataset) in each mini-batch, and (ii) class-weighted categorical cross-entropy with weights $\propto 1/\sqrt{n_k}$ normalised to sum to $K$. QWK is chosen as the checkpoint-selection metric because it is sensitive to ordinal distance and relatively robust to the marginal distribution.

### 3.3 Variables and Operational Definitions

This study involves two independent variables, three dependent variables, and a set of controlled variables, defined operationally in Table 3.1.

\begin{table}[htbp]
\caption{Research variables and their operational definitions.}
\label{tab:variabel}
\begin{center}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|p{3.1cm}|p{2.5cm}|p{7.2cm}|}
\hline
\textbf{Variable} & \textbf{Type} & \textbf{Operational definition} \\
\hline
Preprocessing technique (independent) & Categorical, 5 levels & Intensity or colour transform applied to each image after cropping and before resizing; levels and parameters in Section 3.4. \\
\hline
Backbone architecture (independent) & Categorical, 2 levels & Feature-extraction network (ResNet-50 or ViT-B/16) initialised from ImageNet weights and fine-tuned on the DR grading data; Section 3.5. \\
\hline
Classification performance (dependent) & Continuous & Accuracy, QWK, and macro-F1 on the test partition (Section 2.5). \\
\hline
Cross-dataset consistency (dependent) & Ordinal/Continuous & Average rank of each configuration over the six datasets and its significance under the Friedman test, the Wilcoxon signed-rank test, and per-dataset McNemar tests, as a measure of how consistently a configuration performs across datasets (Section 3.7). \\
\hline
Computational cost (dependent) & Continuous & Number of model parameters and mean inference time per image. \\
\hline
Controls & Fixed & Basic pipeline (Section 3.4.1), augmentation (Section 3.4.3), optimiser and hyperparameters (Section 3.6), data split, random seed, and computing environment (Section 3.8), held identical across configurations. \\
\hline
\end{tabular}
\end{center}\end{table}

### 3.4 Data Preprocessing

#### 3.4.1 Basic Pipeline

Ahead of any technique-specific step, every image across the six datasets goes through one shared basic pipeline: (i) the retinal mask is estimated by adaptive thresholding on the green channel and then cropped to its minimum bounding box to strip away the black frame; (ii) the image is resized to $224 \times 224$ pixels by bilinear interpolation to match the input expected by ImageNet weights; and (iii) values are scaled to $[0, 1]$ and then standardised per channel with the ImageNet mean and standard deviation ($\mu = (0.485, 0.456, 0.406)$, $\sigma = (0.229, 0.224, 0.225)$). The same pipeline governs training, validation, and testing on every dataset, so that no configuration difference is hidden behind a difference in basic preprocessing.

#### 3.4.2 Application of the Five Preprocessing Techniques

The preprocessing technique is applied between cropping and resizing; the parameters of each technique (concepts in Section 2.3) are as follows. **CLAHE**: clip limit 2.0 and tile $8 \times 8$ on the L channel (LAB). **Ben Graham Normalization**: $\alpha = 4$, a Gaussian radius $\sigma$ equal to 10% of the retinal diameter, and $\gamma = 128$, following the original Kaggle 2015 recipe. **Adaptive Sigmoid Enhancement**: $\alpha$ and $\beta$ computed adaptively from the mean and standard deviation of the local patch intensity. **LAB-ACE**: CLAHE on the L channel followed by reconstruction to RGB. **MCIE**: a combination of the original green channel, the CLAHE result on the L channel, and the Ben Graham Normalization result into a three-channel image.

#### 3.4.3 Data Augmentation

Online augmentation is applied only to the training partition of each dataset, after preprocessing. Geometric augmentation consists of horizontal and vertical flips (probability 0.5) and random rotation $[-30^{\circ}, +30^{\circ}]$, which preserve the label because the ICDR grade does not depend on spatial orientation. Photometric augmentation consists of random brightness and contrast jitter $[-0.2, +0.2]$ to simulate the illumination variation across locations. The validation and test partitions of every dataset are not augmented.

### 3.5 Model Architecture

The concepts of both backbones and their architecture diagrams have been described in Section 2.4 (Figures 2.2 and 2.3); this section details only their experimental configuration. ResNet-50 is loaded with ImageNet-1k pretrained weights via torchvision 0.18, while ViT-B/16 is loaded with ImageNet-21k pretrained weights via timm 1.0. In both backbones, the original classification head is replaced with a new linear layer to the five ICDR logits (ResNet-50: $2048 \to 5$ from the global average pooling vector; ViT-B/16: $768 \to 5$ from the CLS token). All parameters are trained jointly (full fine-tuning) without freezing any layer, with a softmax output optimised using class-weighted categorical cross-entropy (Sections 3.2.2 and 3.6).

### 3.6 Training Protocol

Every model is optimised with AdamW (Loshchilov and Hutter, 2019) at an initial learning rate $\eta_{0} = 1 \times 10^{-4}$, weight decay $1 \times 10^{-4}$, $\beta_{1} = 0.9$, and $\beta_{2} = 0.999$. Following a three-epoch linear warm-up, the learning rate decays on a cosine schedule to $\eta_{\min} = 1 \times 10^{-6}$. Batches hold 16 images, which fits within the memory of a single mid-range GPU (about 16 GB). Each run lasts at most 50 epochs and halts early on validation QWK (patience of 10 epochs), and the checkpoint with the best validation QWK is the one carried forward to test evaluation.

The loss function is class-weighted categorical cross-entropy (weights as in Section 3.2.2). Mixup, label smoothing, and focal loss are deliberately not used, so that performance differences between configurations arise purely from the preprocessing technique and the backbone choice. All random number generators are initialised with the same seed so that the results can be reproduced. Because the design trains every configuration on every dataset, there are ten configurations across six datasets, giving sixty training runs in total. To keep this tractable, the very large EyePACS dataset is trained on a class-stratified subset (Section 3.2.1), and the smaller datasets contribute most of the runs at low cost; the indicative total training budget is therefore on the order of a hundred GPU-hours on a single GPU, scheduled across the training months in the work plan (Section 3.9).

### 3.7 Evaluation Protocol

On each of the six datasets, the test partition is evaluated once per configuration with accuracy, QWK, macro-F1, and a per-class confusion matrix, all reported in a per-dataset results table. This produces a $10 \times 6$ matrix of QWK scores, one per configuration per dataset, which is the basis of the statistical analysis.

The statistical analysis proceeds in three stages: an omnibus test, a descriptive ranking, and confirmatory pairwise tests on the leading configurations. As the omnibus stage, the ten configurations are compared across datasets with the Friedman test (Demsar, 2006), the standard non-parametric procedure for comparing several methods over multiple datasets. Within each dataset the ten configurations are ranked by QWK, and the Friedman test examines whether their average ranks differ significantly overall,

$$\chi_F^2 = \frac{12N}{k(k+1)}\left[\sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4}\right],$$

where $N=6$ datasets, $k=10$ configurations, and $R_j$ is the average rank of configuration $j$. QWK is the metric on which the test is run because it is the primary, ordinal-aware metric for DR grading; accuracy and macro-F1 are reported descriptively alongside it. When the Friedman test is significant, the average-rank ordering of the ten configurations is reported descriptively to show which configurations perform best across datasets. Because six datasets give the omnibus test limited power to resolve all ten configurations, this ranking is treated as descriptive evidence of consistency rather than as a set of pairwise significance claims, and the confirmatory testing below is restricted to the two leading configurations.

The two leading configurations (those with the best average rank across datasets) are then compared directly at two levels. Across datasets, they are compared with the Wilcoxon signed-rank test (Wilcoxon, 1945) on their six paired per-dataset QWK values,

$$W = \min(W^+, W^-), \qquad W^+ = \sum_{d_i > 0} \operatorname{rank}(|d_i|),$$

where $d_i$ is the QWK difference between the two configurations on dataset $i$ and ranks are taken over $|d_i|$; the null hypothesis is that the median difference is zero. This is a single planned comparison of two methods, for which six paired datasets provide adequate power. Within each dataset, the same two configurations are compared with McNemar's test (McNemar, 1947) on the paired per-image predictions of that dataset's test partition,

$$\chi^2 = \frac{(b - c)^2}{b + c},$$

where $b$ and $c$ count the images that one configuration classifies correctly and the other does not; operating on thousands of paired samples, this test has substantially higher power within a single dataset. Because it is applied once per dataset, the family of six per-dataset McNemar tests is corrected with the Holm-Bonferroni procedure (Holm, 1979) to control the family-wise error rate.

Beyond comparing whole configurations, the factorial design also allows each factor to be tested separately, which maps directly onto RQ1 and keeps the number of comparisons small enough for six datasets to retain usable power. For the backbone factor, the per-dataset QWK scores are averaged over the five preprocessing techniques to give one value per backbone on each dataset, and ResNet-50 and ViT-B/16 are then compared across the six datasets with the Wilcoxon signed-rank test, a single planned comparison. For the preprocessing factor, the per-dataset QWK scores are averaged over the two backbones to give a $5 \times 6$ matrix, on which the five preprocessing techniques are compared with the Friedman test; when it is significant, each technique is compared against the baseline with pairwise Wilcoxon signed-rank tests under Holm-Bonferroni correction. The preprocessing $\times$ backbone interaction, that is, whether the best preprocessing technique differs between the two backbones, is examined descriptively by comparing the preprocessing ranking within each backbone separately.

The division of labour is explicit: the across-dataset Wilcoxon test answers whether the leading configuration is consistently better over the six datasets, whereas the within-dataset McNemar tests answer whether the two configurations differ on a given dataset. A configuration is regarded as robustly superior only when it both leads the cross-dataset ranking and wins the within-dataset McNemar comparisons. Following the principle of pre-specifying outcomes, if no configuration separates from the field, the conclusion that the best configuration is dataset-specific is reported as a substantive result consistent with the study's motivation (Section 1.1), not as an inconclusive finding.

The analysis is implemented with the \texttt{scipy.stats} and \texttt{statsmodels} libraries (Friedman and Wilcoxon signed-rank tests, McNemar's test, and the Holm-Bonferroni correction). In addition, Grad-CAM (Selvaraju et al., 2017) is generated on the best configuration for visual verification of the basis of the predictions.

### 3.8 Implementation Tools and Environment

The experimental pipeline is implemented in Python 3.11 using the following libraries: PyTorch 2.3 as the tensor and autodiff backend; torchvision 0.18 for ResNet-50 and the ImageNet-1k pretrained weights; timm 1.0 (Wightman, 2019) for ViT-B/16 and the ImageNet-21k pretrained weights; OpenCV 4.9 and scikit-image 0.22 for implementing the five preprocessing techniques; Albumentations 1.4 for the augmentation pipeline; scikit-learn 1.4 for computing the evaluation metrics; SciPy 1.13 and statsmodels 0.14 for the statistical tests (Friedman, Wilcoxon signed-rank, McNemar, and the Holm-Bonferroni correction); and Matplotlib 3.8 and seaborn 0.13 for visualising the confusion matrices and Grad-CAM heatmaps. The experiments are run in a single-GPU environment; a mid-range GPU with roughly 16 GB of memory is sufficient for the chosen batch size and input resolution, and the exact hardware may vary with the computational resources available at the time of training. All code, training scripts, evaluation scripts, configuration files, and metric logs are stored in a public Git repository accompanied by a reproduction README with explicit library versions, seeds, and execution commands.

### 3.9 Research Schedule

The research activities are planned to run for twelve months, from March 2026 to February 2027, covering literature study and proposal writing, the proposal seminar and its revision, environment preparation and collection of the six datasets, implementation of the preprocessing pipeline, training of the ten configurations on each of the six datasets, per-dataset evaluation, Friedman statistical analysis and Grad-CAM visualisation, the writing of Chapter IV and Chapter V, the results seminar, and the thesis defense and final submission. The monthly schedule is detailed in Table 3.2; the columns from March to December are months in 2026, while January and February are months in 2027. The schedule is indicative and may be adjusted according to the results of consultations with the supervisors and the availability of computational resources.

\begin{table}[htbp]
\caption{Research schedule for the period March 2026 to February 2027.}
\label{tab:jadwal}
\begin{center}
\small
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{|c|p{3.3cm}|c|c|c|c|c|c|c|c|c|c|c|c|}
\hline
\textbf{No} & \textbf{Activity} & \textbf{Mar} & \textbf{Apr} & \textbf{May} & \textbf{Jun} & \textbf{Jul} & \textbf{Aug} & \textbf{Sep} & \textbf{Oct} & \textbf{Nov} & \textbf{Dec} & \textbf{Jan} & \textbf{Feb} \\
\hline
1 & Literature study and proposal writing & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  &  &  \\
\hline
2 & Proposal seminar and revision &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  &  \\
\hline
3 & \emph{Dataset} and environment preparation &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  &  \\
\hline
4 & \emph{Preprocessing} implementation (5 techniques) &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  &  &  \\
\hline
5 & Training 10 configs on 6 datasets &  &  &  &  & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} &  &  &  &  &  \\
\hline
6 & Per-\emph{dataset} evaluation &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  &  \\
\hline
7 & Friedman analysis and Grad-CAM &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  &  \\
\hline
8 & Writing Chapter IV &  &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  &  \\
\hline
9 & Writing Chapter V and revision &  &  &  &  &  &  &  &  &  & \cellcolor{black} & \cellcolor{black} &  \\
\hline
10 & Supervision (Supervisor I and II) & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} & \cellcolor{black} &  \\
\hline
11 & Results seminar &  &  &  &  &  &  &  &  &  &  & \cellcolor{black} &  \\
\hline
12 & Thesis defense and final submission &  &  &  &  &  &  &  &  &  &  &  & \cellcolor{black} \\
\hline
\end{tabular}
\end{center}\end{table}



## REFERENCES {.unnumbered}

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
