# THESIS PROPOSAL

**Title:**
Comparative Analysis of Fundus Image Preprocessing Techniques and Deep Learning Architectures for Diabetic Retinopathy Severity Classification across Multiple Public Datasets

---

## INTRODUCTION

### 1.1 Research Background

Diabetic retinopathy (DR), a progressive microvascular complication of diabetes mellitus, is the foremost preventable cause of blindness in the working-age population (American Diabetes Association, 2024), and the problem is widening quickly: Teo et al. (2021) project that the 103 million adults affected in 2020 will reach roughly 161 million by 2045. Indonesia confronts this acutely. A DR incidence of 34.6 per 1,000 person-years (Sasongko et al., 2025) and a 55% prevalence among diabetic patients at a referral hospital (Saputra et al., 2024) meet an ophthalmologist supply of fewer than two per 100,000 population clustered in urban Java, making the yearly fundus examination that guidelines call for impossible to deliver by hand at scale. Automating fundus-image interpretation in primary care is the only workable response, and the clinically relevant task is not merely flagging disease but assigning a severity grade on the five-level International Clinical DR (ICDR) scale (Wilkinson et al., 2003), since that grade decides referral. Errors are costly either way: too low a grade delays a needed referral, too high a grade consumes scarce specialist time, so accuracy at the grade boundaries is what counts.

Deep learning has turned automated DR grading from an aspiration into a working reality over the last ten years. Using a convolutional neural network trained on 128,175 fundus images, Gulshan et al. (2016) reported sensitivity and specificity above 90% for referable DR; Abramoff et al. (2018) subsequently secured FDA clearance for IDx-DR, the first autonomous DR system; and foundation models such as RETFound (Zhou et al., 2023) have since closed much of the remaining distance. Surveying more than fifty studies and twenty datasets, Chopra et al. (2025) confirm this maturity yet observe that the central challenge has moved away from raw capability and toward multi-center validation and clinical trust.

The first neglected gap lies in image quality and its treatment. Inconsistent illumination, weak contrast, and capture noise erode grading accuracy in a systematic way (Anupama et al., 2025), an effect that is worst on the inexpensive cameras typical of Indonesian primary care; because such degradation hides early lesions like microaneurysms, it pushes the model toward grades that fall below the patient's true condition. The literature offers many preprocessing methods for making lesions more visible, ranging from CLAHE, Ben Graham preprocessing, and green-channel extraction to more recent proposals such as Adaptive Sigmoid Enhancement, LAB-ACE, and Multi-channel Image Enhancement, yet they are adopted in an ad-hoc fashion, with each study committing to one method and never weighing it against the others under a shared architecture and protocol. Anupama et al. (2025) note the same problem and ask for a systematic study of preprocessing options, since without a like-for-like comparison the decision rests on guesswork instead of evidence.

The second gap concerns how far these comparisons generalise. Nearly every preprocessing comparison for DR grading, the anchor study of Anupama et al. (2025) included, is confirmed on just one dataset, so whichever technique or backbone comes out ahead there may reflect that dataset's particular cameras, population, and quality rather than a property that travels. Since fundus images differ markedly in illumination, colour, and acquisition protocol from one institution to the next, a method that prevails on one dataset will not necessarily prevail on another, and a recommendation built on a single dataset is a fragile basis for a screening system meant to face field conditions that vary widely. Anupama et al. (2025) therefore advise testing findings across several datasets that span different devices and demographics; until that is done, the external validity of any comparison of this kind stays unproven. Taken together, these two gaps leave an empty cell in the literature: no prior study has compared a set of preprocessing techniques head-to-head, across more than one backbone paradigm, over several datasets, and with a formal statistical test of whether the resulting verdict holds across them (Section 2.6).

This study is designed to fill that cell, and in doing so takes up the future-work agenda of Anupama et al. (2025), tackling both gaps together. Five preprocessing techniques are paired with two backbones (ResNet-50 and ViT-B/16) to give ten configurations, which are assessed separately on six public DR grading datasets built on the same five-grade ICDR ontology (IDRiD, DDR, APTOS 2019, Messidor-2, EyePACS, and DeepDRiD); within each dataset, every configuration is trained and tested on that dataset's own partition under one common protocol and scored by accuracy, quadratic-weighted kappa (QWK), and macro-F1. To locate the configuration that excels not on a single dataset but uniformly across all six, the per-dataset QWK values are combined through the Friedman test (Demsar, 2006) and the leading configurations are compared with the Wilcoxon signed-rank test across datasets and with per-dataset McNemar tests, while Grad-CAM checks what the leading configuration's predictions actually rest on. The contributions are threefold: (i) a reproducible, like-for-like comparison of preprocessing methods and backbones for DR grading under one unified protocol; (ii) a formal, cross-dataset assessment of that comparison, aggregating the per-dataset results with non-parametric statistical tests (the Friedman, Wilcoxon signed-rank, and McNemar tests) to establish whether the best configuration is genuinely stable across datasets rather than tied to a particular one, which is the methodological step that separates this work from earlier single-dataset comparisons; and (iii) evidence-based guidance for building automated DR screening in resource-limited primary care.

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

On the theoretical side, the study yields a reproducible empirical comparison of preprocessing techniques against backbone architectures for DR grading under one unified protocol, reinforced by running that comparison over six public datasets that share a single label ontology yet differ in device, population, and illumination. Aggregating the per-dataset results across datasets (Section 3.7) lets the study determine whether the leading configuration holds up across datasets or merely reflects one of them, which speaks directly to the future-work agenda of Anupama et al. (2025).

On the practical side, the study supplies evidence-based guidance for building automated DR screening in Indonesian primary care, a setting whose devices and populations vary widely in the field. The preprocessing-and-backbone pairing that proves consistently strongest across diverse datasets can serve as a default starting point for development. The work additionally delivers a documented experimental pipeline that can be reused to evaluate further preprocessing techniques, backbones, or datasets without rebuilding it from scratch.

## THEORETICAL FRAMEWORK

### 2.1 Diabetic Retinopathy

#### 2.1.1 Brief Pathophysiology

Diabetic retinopathy is a chronic microvascular consequence of diabetes mellitus. Sustained hyperglycaemia injures the retinal capillaries, which both become more permeable (producing oedema and lipid exudation) and occlude (producing ischaemia); that ischaemia in turn stimulates vascular endothelial growth factor (VEGF) and the growth of new vessels (Wong and Sabanayagam, 2023). The disease is split clinically into two stages. The non-proliferative stage (NPDR) presents microaneurysms, hard exudates, cotton-wool spots, intraretinal haemorrhages, venous beading, and intraretinal microvascular abnormalities (IRMA) but no new vessels, whereas the proliferative stage (PDR) is defined by neovascularisation, which can progress to vitreous haemorrhage and tractional retinal detachment. Which lesions appear, and where they sit, is what underpins the ICDR severity scale (Section 2.1.3) and is precisely what the Grad-CAM check targets (Section 2.5.3).

#### 2.1.2 Global and Indonesian Disease Burden

As detailed in Section 1.1, DR is a leading cause of preventable blindness whose global burden is projected to reach about 161 million people by 2045 (Teo et al., 2021), a trajectory Wong and Sabanayagam (2023) call the "DR pandemic". In Indonesia this is compounded by a severe shortage of ophthalmologists (fewer than two per 100,000, concentrated in urban Java), the structural reason automated DR screening at the primary-care level is needed (Sasongko et al., 2025; Saputra et al., 2024; American Diabetes Association, 2024).

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

A concern that naturally follows is whether a given ICDR grade means the same thing across datasets graded by different people: is a grade 1 (mild) in the Indian IDRiD the same as a grade 1 in the Chinese DDR? Two points address this. First, the grades are not idiosyncratic per-dataset scales but are all defined against a single published clinical standard, the ICDR severity scale of Wilkinson et al. (2003), which fixes what each grade means (for instance, grade 1 is mild NPDR with microaneurysms only, and grade 3 follows the 4-2-1 rule set out in Section 2.1.3). The label semantics are therefore shared by construction, even where the graders differ. Second, several of the datasets go beyond single-grader labelling and use adjudicated reference standards: the Messidor-2 grades used here were produced by the adjudication protocol of Krause et al. (2018), and the DeepDRiD grades come from adjudication among several ophthalmologists, an approach that Krause et al. (2018) show reduces grader variability and yields higher-quality labels than single readings. Grader variability nonetheless remains a known property of DR datasets, more pronounced for large single-grader sources such as EyePACS, and the study does not claim it away.

Crucially, the experimental design is robust to whatever inter-dataset differences remain, because it never pools the datasets. Each of the ten configurations is trained and tested on a single dataset against that dataset's own labels, and the cross-dataset analysis is rank-based (Section 3.7): within each dataset the ten configurations are ranked on that dataset's own labels, and only those ranks are aggregated across datasets with the Friedman and Wilcoxon tests. A systematic difference in grading strictness between two datasets therefore shifts all ten configurations on a dataset by the same amount and cancels out of the ranking; one dataset's labels can never contaminate another dataset's training. The study thus does not compare grades across datasets or assume that grades are equivalent between them; it asks only which configuration ranks most consistently when each dataset is judged on its own terms. Should any single dataset nonetheless prove anomalous in practice, the same design permits reporting the comparison on the largest and most reliably graded datasets (for example EyePACS) alone, without changing the protocol.

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

#### 2.3.6 Rationale for Selecting the Five Techniques

The five techniques are not an arbitrary selection; they fall into two groups, summarised in Table 2.1. The first group holds the two preprocessing steps most widely adopted in DR grading. Ben Graham Normalization (Graham, 2015) is the method that won the Kaggle Diabetic Retinopathy Detection competition and has since become a de-facto default, while CLAHE (Zuiderveld, 1994) recurs across DR pipelines and is used by Anupama et al. (2025) themselves as a component of their composite pipeline, chosen there to raise local contrast so that microaneurysms become easier to see. These two are the established baselines against which any newer method must prove itself.

The second group holds the three techniques introduced by Anupama et al. (2025), the paper this study builds on. On the APTOS 2019 dataset they reported Adaptive Sigmoid Enhancement as their strongest pipeline (96.39\% accuracy using ResNet-50 features and an XGBoost classifier), followed by LAB-ACE (91.84\%) and MCIE (86.62\%). Two features of that work motivate the present design. First, the three techniques were assessed only within Anupama et al.'s (2025) own multi-modal fusion pipeline and against one another, not benchmarked like-for-like against the established methods across different backbones; the authors themselves flag a component-level study as a direction to pursue and put MCIE-style multi-channel combinations forward as promising. Second, because the evaluation rested on the single APTOS 2019 dataset, it is unknown whether the ranking transfers, which is why the same authors list multi-center validation among their future-work directions. This study takes up both points: it places the three novel techniques head-to-head with the two field standards under one uniform protocol and across six datasets, so the comparison is like-for-like rather than tied to a single pipeline and a single dataset. Across the set, the five techniques also span the main mechanisms found in the literature, from local-contrast enhancement (CLAHE, LAB-ACE) through global-illumination correction (Ben Graham) and adaptive intensity stretching (Adaptive Sigmoid) to multi-representation compositing (MCIE), so that they interact differently with each backbone's inductive bias.

\begin{table}[htbp]
\caption{The five preprocessing techniques and the rationale for their selection. The accuracy figures are those reported by Anupama et al. (2025) on APTOS 2019 with ResNet-50 features and an XGBoost classifier.}
\label{tab:prepselection}
\begin{center}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|p{2.8cm}|p{3.4cm}|p{2.9cm}|p{3.4cm}|}
\hline
\textbf{Technique} & \textbf{Origin / representative study} & \textbf{What it standardises} & \textbf{Role in this study} \\
\hline
CLAHE & Zuiderveld (1994); used in DR by Anupama et al. (2025) & Local contrast (luminance) & Field-standard baseline \\
\hline
Ben Graham Normalization & Graham (2015), Kaggle DR winner & Global illumination & Field-standard baseline \\
\hline
Adaptive Sigmoid Enhancement & Anupama et al. (2025) & Adaptive intensity and contrast & Best novelty of the baseline paper (96.39\%) \\
\hline
LAB-ACE & Anupama et al. (2025) & Lesion contrast without colour shift & Baseline-paper novelty (91.84\%) \\
\hline
MCIE & Anupama et al. (2025) & Multi-representation composite & Baseline-paper novelty (86.62\%), proposed as a future direction \\
\hline
\end{tabular}
\end{center}\end{table}

### 2.4 Deep Learning Architectures for Image Classification

This study evaluates two deep learning backbones representing two different paradigms, namely the convolutional neural network (CNN) represented by ResNet-50 and the Vision Transformer (ViT) represented by ViT-B/16. Neither is chosen arbitrarily. ResNet-50 is the very network Anupama et al. (2025) used as the deep feature extractor in their fusion pipeline, and it recurs as a standard baseline across the DR grading literature, so retaining it keeps this study directly comparable to the baseline paper. ViT-B/16 is chosen because Anupama et al. (2025), among their future-work directions, name Vision Transformers explicitly, for modelling long-range dependencies and global context in fundus images, as a direction to pursue; that transformers transfer well to retinal images is already evidenced by ViT-based foundation models such as RETFound (Zhou et al., 2023). Pairing a CNN with a ViT turns the question of which backbone to use from an untested assumption into a controlled variable, which is what lets the study separate the effect of preprocessing from the effect of the backbone. The Base rather than the Large ViT is used so that the model stays within the same order of magnitude and compute budget as ResNet-50 (about 25.5 million parameters) and fits a single mid-range GPU of about 16 GB, keeping the comparison both fair and feasible; Anupama et al. (2025) recommend ViT in general terms without prescribing a scale.

These choices are backed by how the two paradigms have actually performed on the kind of data used here. ResNet-50 is not merely conventional: it is the backbone Chokuwa and Khan (2025) adopt for domain-generalization experiments across seven DR grading datasets, the CNN whose features drive Anupama et al.'s (2025) best pipeline on APTOS 2019, and the network behind Kumar et al.'s (2025) ordinal-regression model that reaches a QWK of 0.899 on APTOS 2019; it recurs as the strong, reproducible baseline on exactly this class of data, which is why it anchors the CNN side of the comparison. On the Transformer side, the case for ViT rests on capability rather than novelty. Its self-attention models relationships across the whole image, which suits DR grading, where the lesions that set the grade (microaneurysms, haemorrhages, neovascularisation) can lie in different retinal quadrants; and ViT-based models have already delivered leading results on retinal images, with RETFound (Zhou et al., 2023) attaining state-of-the-art disease detection on fundus photographs including DR. ViT-B/16 is therefore expected to be genuinely competitive with ResNet-50 rather than a placeholder, and the study is designed to measure that head to head.

A reasonable follow-up is why ViT-B/16 rather than a later Transformer such as the Swin Transformer, whose hierarchical local windows were introduced to address ViT's limitations, and which Mok et al. (2024) indeed use as the backbone for referable-DR classification. Swin is not adopted here for two reasons: its hierarchical, higher-resolution design is markedly heavier to train under the single mid-range GPU budget of this study, and its parameter count and scale do not line up with ResNet-50, which would confound the like-for-like CNN-versus-Transformer contrast that motivates the pairing. ViT-B/16 keeps that contrast clean while remaining feasible. Each architecture is described in Sections 2.4.1 and 2.4.2.

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

Since ICDR is ordinal, an error of two grades is worse than an error of one, a distinction that accuracy and F1 ignore. Quadratic-weighted kappa (Cohen, 1968) applies a penalty that grows with the square of the gap between classes, as defined in Equation 2.1,

\begin{equation}
\kappa_{w} = 1 - \frac{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} O_{ij}}{\sum_{i=1}^{K} \sum_{j=1}^{K} w_{ij} E_{ij}},
\qquad w_{ij} = \frac{(i - j)^{2}}{(K - 1)^{2}},
\label{eq:qwk}
\end{equation}

in which $O$ is the observed confusion matrix and $E$ is the confusion matrix expected were predictions and labels independent. Here $\kappa_w = 1$ signals perfect agreement and $\kappa_w = 0$ corresponds to chance-level guessing. Established as the standard score in DR grading challenges (Kaggle Diabetic Retinopathy Detection, APTOS 2019), QWK is the primary metric of this study.

#### 2.5.3 Gradient-weighted Class Activation Mapping (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) yields a heatmap of the regions that most sway the classification, obtained by weighting the final convolutional layer's feature maps with the gradient of the target class score. For DR grading, it helps confirm whether a prediction is grounded in clinically meaningful lesions (such as microaneurysms or haemorrhages) rather than in artefacts like the lens edge. The present study treats Grad-CAM as a supporting interpretation tool applied to the best configuration rather than as the main quantitative concern, in keeping with the recommendation of Anupama et al. (2025).

### 2.6 Related Work

As Section 1.1 noted, deep-learning DR grading attained clinical and regulatory maturity through landmark systems (Gulshan et al., 2016; Abramoff et al., 2018); the subsequent release of public benchmarks such as IDRiD (Porwal et al., 2020) and DDR (Li et al., 2019) then turned attention to openly comparing backbones (ResNet, DenseNet, Inception, EfficientNet) and, more recently, foundation models such as RETFound (Zhou et al., 2023). The present study belongs to this open-benchmark tradition but concentrates on two questions that remain underexplored: how preprocessing and backbone interact, and whether the comparison that results carries over across datasets.

Chokuwa and Khan (2025) demonstrated that DR grading models lose substantial performance when one trained on a given dataset is tested on another whose acquisition settings differ, the cause being domain shift (camera, illumination, demographics) rather than inconsistent annotation. That result is why this study declines to move a single model between datasets and instead treats each dataset as a self-contained benchmark, asking whether the comparative verdict, that is, which preprocessing and backbone come out best, stays the same across datasets.

This study's anchor paper is Anupama et al. (2025) in Scientific Reports, which assesses several backbones for DR grading on one dataset and, in its future-work section, points to three directions: (i) a systematic study of preprocessing combinations; (ii) validation on multi-center datasets covering diverse devices and demographics; and (iii) the addition of interpretation tools such as Grad-CAM. The present work builds on all three, comparing five preprocessing techniques (Section 2.3) across two backbones from different paradigms (Section 2.4), evaluating them independently over six public datasets (Section 2.2), and applying Grad-CAM to the best configuration (Section 2.5.3).

To make the study's position explicit, Table 2.2 situates it among representative deep-learning DR works of the past decade, compared on the axes that bear on its research questions: whether preprocessing techniques are benchmarked like-for-like under one protocol, which backbone is used, how many datasets are involved, and whether a formal test compares methods across datasets. Two gaps stand out. First, no prior study compares a set of preprocessing techniques head-to-head under one fixed protocol: preprocessing is either a single fixed pipeline (Tusfiqur et al., 2022; Chokuwa and Khan, 2025; Kumar et al., 2025) or, in the anchor paper, a set of novel methods assessed only within one fusion pipeline on one dataset (Anupama et al., 2025). Second, the evaluations that do span several datasets (Zhou et al., 2023; Chokuwa and Khan, 2025) do so to build or stress-test a single model rather than to ask whether a preprocessing-and-backbone verdict is stable, and only RETFound reports a cross-dataset significance test. This study fills the empty cell in Table 2.2: a like-for-like comparison of five preprocessing techniques across two backbone paradigms (CNN and ViT), evaluated on six datasets and aggregated with formal non-parametric tests (Friedman, Wilcoxon signed-rank, and McNemar).

\begin{table}[htbp]
\caption{Positioning of this study among representative deep-learning DR works. "Preproc. compared" indicates whether several preprocessing techniques are benchmarked like-for-like under one protocol; "Cross-dataset test" indicates a formal significance test used to compare methods across datasets.}
\label{tab:relatedwork}
\begin{center}
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|p{2.5cm}|p{2.0cm}|p{1.5cm}|p{2.3cm}|c|p{2.4cm}|}
\hline
\textbf{Study (Year)} & \textbf{Task} & \textbf{Preproc. compared} & \textbf{Backbone(s)} & \textbf{Data-sets} & \textbf{Cross-dataset test} \\
\hline
Gulshan et al. (2016) & Referable DR detection & No & Inception-v3 & 2 & No \\
\hline
Abramoff et al. (2018) & Autonomous DR detection & No & CNN (IDx-DR) & 1 & No \\
\hline
Tusfiqur et al. (2022), DRG-Net & Lesion seg. $+$ grading & No & ResNet-50, ViT & 3 & No \\
\hline
Zhou et al. (2023), RETFound & Foundation model (incl. DR) & No & ViT (MAE) & 3 & t-test (per task) \\
\hline
Mok et al. (2024) & Referable DR $+$ lesion maps & No & Swin Transformer & 2 & No \\
\hline
Chokuwa \& Khan (2025) & Grading (domain gen.) & No & ResNet-50 & 7 & No \\
\hline
Anupama et al. (2025) (anchor) & Grading $+$ fusion & Partly & ResNet-50 $+$ XGBoost & 1 & No \\
\hline
Kumar et al. (2025) & Grading (ordinal) & No & ResNet-50 & 1 & No \\
\hline
\textbf{This study (2026)} & Grading (5-class) & \textbf{Yes (5)} & ResNet-50 $+$ ViT-B/16 & \textbf{6} & \textbf{Friedman, Wilcoxon, McNemar} \\
\hline
\end{tabular}
\end{center}\end{table}

### 2.7 Conceptual Framework

Raw fundus images vary in illumination, contrast, and colour, and the earliest DR lesions are small and low in contrast (Sections 2.1 and 2.2). Because the five preprocessing techniques (Section 2.3) and the two backbones (Section 2.4) act through different mechanisms and inductive biases, performance is viewed as a function of the preprocessing $\times$ backbone interaction, which the $5 \times 2$ factorial design (Section 3.1) is built to isolate. To test whether that interaction holds beyond a single dataset, the ten configurations are trained and evaluated independently on six datasets that differ in camera, population, and illumination yet share the ICDR ontology, so that a configuration ranking first on all six can claim external validity. Because the ICDR scale is ordinal and imbalanced, the comparison relies on QWK and macro-F1 rather than accuracy alone (Section 2.5) and aggregates the per-dataset QWK with the tests detailed in Section 3.7. These considerations lead to the hypotheses in Section 2.8.

### 2.8 Research Hypotheses

Following the research questions, this study is guided by four statistical hypotheses, each mapped to one aspect of the $5 \times 2$ design and to the test that evaluates it in Section 3.7. Throughout, a *configuration* denotes one (preprocessing technique, backbone) pair, so there are $5 \times 2 = 10$ configurations, and QWK is the metric on which the hypotheses are tested. The first two hypotheses isolate each factor, the third addresses their interaction, and the fourth compares the two leading configurations directly; together they answer RQ1 (H1 to H3) and RQ2 (H4).

**Hypothesis 1 (effect of preprocessing).** Averaging QWK over the two backbones to give one value per technique on each dataset:

- $H1_0$: across the six datasets the five preprocessing techniques (CLAHE, Ben Graham Normalization, Adaptive Sigmoid Enhancement, LAB-ACE, and MCIE) have equal median QWK; the choice of preprocessing has no systematic effect.
- $H1_1$: at least one preprocessing technique has a different median QWK; the choice of preprocessing systematically affects grading performance across datasets.
- Tested with the Friedman test over the resulting $5 \times 6$ matrix, followed, when significant, by pairwise Wilcoxon signed-rank tests against the baseline under Holm-Bonferroni correction.

**Hypothesis 2 (effect of backbone).** Averaging QWK over the five preprocessing techniques to give one value per backbone on each dataset:

- $H2_0$: across the six datasets the median difference in QWK between ResNet-50 and ViT-B/16 is zero; neither backbone consistently outperforms the other.
- $H2_1$: that median difference is not zero; one backbone systematically outperforms the other across datasets.
- Tested with the Wilcoxon signed-rank test on the six paired per-dataset QWK values.

**Hypothesis 3 (preprocessing $\times$ backbone interaction).**

- $H3_0$: the best-performing preprocessing technique is the same for both backbones; the ranking of preprocessing techniques is consistent across ResNet-50 and ViT-B/16 (no interaction).
- $H3_1$: the best preprocessing technique differs between the two backbones; the effect of preprocessing depends on the backbone (an interaction is present).
- Examined descriptively by comparing the preprocessing ranking within each backbone separately, because six datasets give a formal interaction test little power (Section 3.7).

**Hypothesis 4 (comparison of the two leading configurations).** Let the two leading configurations be those with the best average Friedman rank across datasets:

- $H4_0$: across the six datasets the median difference in their QWK is zero, and within each dataset their paired per-image predictions do not differ.
- $H4_1$: the two configurations differ, one attaining a higher median QWK across datasets or a significant within-dataset difference.
- Tested across datasets with the Wilcoxon signed-rank test and within each dataset with Holm-Bonferroni-corrected McNemar tests; this is what decides whether a numerically small gap (for example 0.80 versus 0.82 in QWK) is statistically real.

All four hypotheses are framed on medians and ranks rather than means, because they are tested with non-parametric procedures, the Friedman and Wilcoxon signed-rank tests, which compare the median of the paired per-dataset differences rather than their average. The choice is deliberate and follows the standard methodology for comparing methods over multiple datasets. Demsar (2006), whose work is the reference procedure for exactly this setting, recommends against averaging a performance score across datasets and instead prescribes these rank-based non-parametric tests. Two properties of the present study make that recommendation apply directly. First, a mean assumes that QWK values are commensurable across datasets and is easily dominated by outliers, yet the six datasets differ by more than two orders of magnitude in size (516 to 88,702 images) and in difficulty, so one unusually hard or easy dataset could distort a mean, whereas the median and the ranks the tests use are robust to such outliers. Second, with only six datasets the normality that a mean-based paired t-test assumes cannot be established, whereas these tests make no normality assumption. The median therefore reports the typical cross-dataset advantage of one method over another, which is the quantity of interest for judging consistency.

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

Complementing the design view of Figure 3.1, Figure 3.2 lays out the research as a linear sequence of stages (the research procedure), which is the order that the twelve-month schedule in Section 3.9 follows.

\begin{figure}[htbp]
\centering
\resizebox{\textwidth}{!}{%
\begin{tikzpicture}[
  font=\footnotesize,
  stage/.style={draw, rounded corners=3pt, fill=blue!8, align=center, text width=2.6cm, minimum height=1.5cm, inner sep=3pt, font=\scriptsize},
  arr/.style={-{Stealth[length=2mm]}, semithick}
]
\node[stage] (s1) {1. Literature study and problem formulation};
\node[stage, right=0.5cm of s1] (s2) {2. Six-dataset collection and per-dataset partitioning};
\node[stage, right=0.5cm of s2] (s3) {3. Preprocessing implementation (five techniques)};
\node[stage, right=0.5cm of s3] (s4) {4. Training the ten configurations on each dataset};
\node[stage, right=0.5cm of s4] (s5) {5. Per-dataset evaluation (accuracy, QWK, macro-F1)};
\node[stage, right=0.5cm of s5] (s6) {6. Statistical analysis (Friedman, Wilcoxon, McNemar) and Grad-CAM};
\foreach \a/\b in {s1/s2,s2/s3,s3/s4,s4/s5,s5/s6}{\draw[arr] (\a)--(\b);}
\end{tikzpicture}%
}
\caption{The end-to-end research procedure as a sequence of six stages, from literature study through data preparation, preprocessing, training, and evaluation to statistical analysis and Grad-CAM. Whereas Figure 3.1 shows the logic of the $5 \times 2$ factorial design, this figure shows the chronological workflow that the schedule in Section 3.9 follows.}
\label{fig:stages}
\end{figure}

### 3.2 Dataset

#### 3.2.1 Per-Dataset Train, Validation, and Test Partitions

Each of the six datasets is split into train, validation, and test partitions independently, and all ten configurations are trained and evaluated within each dataset using these partitions. The validation partition is used for early stopping and checkpoint selection; the test partition is accessed only once per configuration, at the end of training, to produce the reported metrics. For datasets that provide official partitions, those are adopted unchanged; for the others, a stratified split under a fixed seed is used.

For **IDRiD**, the official partition (413 training, 103 test; Porwal et al., 2020) is used, with 15% of the training images (62 images) set aside as a stratified validation partition under a fixed seed, leaving 351 training images. For **DDR** (Li et al., 2019), the official train, validation, and test partitions are used after ungradable images are filtered out. For **DeepDRiD** (Liu et al., 2022) and **EyePACS** (Kaggle and EyePACS, 2015), the official partitions are likewise adopted; for EyePACS, ungradable images are removed and a class-stratified subset is drawn to keep the training cost manageable while preserving the class distribution. For **APTOS 2019** (APTOS, 2019) and **Messidor-2** (Decenciere et al., 2014), which do not provide official train and test partitions, a stratified 70/15/15 train/validation/test split under a fixed seed is used. All splits are released in the public repository so that they can be reproduced exactly.

#### 3.2.2 Class Distribution and Handling of Imbalance

Table 3.1 reports, for each dataset, the number of images in every ICDR grade, presented as-is without any rebalancing. Two patterns hold across all six datasets: grade 0 (no DR) is the majority class, and the sight-threatening grades 3 (severe NPDR) and 4 (PDR) are consistently the rarest. The absolute magnitudes differ widely, from 516 images in IDRiD to 88,702 in EyePACS, yet the long-tailed shape recurs regardless of scale, country, or acquisition device.

\begin{table}[htbp]
\caption{Per-class image distribution across the six datasets on the five-grade ICDR scale (0 = No DR, 1 = Mild NPDR, 2 = Moderate NPDR, 3 = Severe NPDR, 4 = PDR). For EyePACS the full published dataset is shown, whereas the study trains on a quality-filtered, class-stratified subset (Section 3.2.1); DeepDRiD is counted at the image level after retaining one field per eye (994 of its 2,000 raw regular images).}
\label{tab:distribution}
\begin{center}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Dataset} & \textbf{0} & \textbf{1} & \textbf{2} & \textbf{3} & \textbf{4} & \textbf{Total} \\
\hline
IDRiD & 168 & 25 & 168 & 93 & 62 & 516 \\
\hline
DeepDRiD & 453 & 111 & 197 & 177 & 56 & 994 \\
\hline
Messidor-2 & 1,017 & 270 & 347 & 75 & 35 & 1,744 \\
\hline
APTOS 2019 & 1,805 & 370 & 999 & 193 & 295 & 3,662 \\
\hline
DDR & 6,266 & 630 & 4,477 & 236 & 913 & 12,522 \\
\hline
EyePACS & 65,343 & 6,205 & 13,153 & 2,087 & 1,914 & 88,702 \\
\hline
\end{tabular}
\end{center}\end{table}

The degree of imbalance itself varies by dataset, as Table 3.2 summarises through the imbalance ratio (IR, the largest class divided by the smallest) and a normalised-entropy balance score (1 = perfectly uniform). IDRiD is the most balanced (IR 6.7, with grades 0 and 2 each about one third of its 516 images), whereas EyePACS is the most skewed (grade 0 alone is 74\% of its images, IR 34.1). Messidor-2, DDR, and EyePACS fall in the highly imbalanced (long-tailed) band, while IDRiD, DeepDRiD, and APTOS 2019 are only moderately imbalanced; no dataset is close to uniform.

\begin{table}[htbp]
\caption{Class-balance summary per dataset. IR is the imbalance ratio (largest class divided by smallest); Majority \% is the share of the largest class; Balance is the normalised Shannon entropy of the class proportions (1 = perfectly uniform).}
\label{tab:balance}
\begin{center}
\small
\renewcommand{\arraystretch}{1.3}
\begin{tabular}{|l|c|c|c|p{3.8cm}|}
\hline
\textbf{Dataset} & \textbf{IR} & \textbf{Majority \%} & \textbf{Balance} & \textbf{Status} \\
\hline
IDRiD & 6.7 & 32.6 & 0.90 & moderately imbalanced \\
\hline
DeepDRiD & 8.1 & 45.6 & 0.87 & moderately imbalanced \\
\hline
Messidor-2 & 29.1 & 58.3 & 0.71 & highly imbalanced (long-tailed) \\
\hline
APTOS 2019 & 9.4 & 49.3 & 0.80 & moderately imbalanced \\
\hline
DDR & 26.6 & 50.0 & 0.70 & highly imbalanced (long-tailed) \\
\hline
EyePACS & 34.1 & 73.7 & 0.54 & highly imbalanced (long-tailed) \\
\hline
\end{tabular}
\end{center}\end{table}

Because this imbalance is intrinsic to the screening population rather than an artefact to be corrected away, it is handled identically across configurations at two levels, namely (i) a weighted random sampler with weights $\propto 1/\sqrt{n_k}$ ($n_k$ is the number of training images of class $k$ in that dataset) in each mini-batch, and (ii) class-weighted categorical cross-entropy with weights $\propto 1/\sqrt{n_k}$ normalised to sum to $K$. QWK is chosen as the checkpoint-selection metric because it is sensitive to ordinal distance and relatively robust to the marginal distribution.

### 3.3 Variables and Operational Definitions

This study involves two independent variables, three dependent variables, and a set of controlled variables, defined operationally in Table 3.3.

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
Cross-dataset consistency (dependent) & Ordinal or continuous & Average rank of each configuration over the six datasets and its significance under the Friedman test, the Wilcoxon signed-rank test, and per-dataset McNemar tests, as a measure of how consistently a configuration performs across datasets (Section 3.7). \\
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

The choice of $224 \times 224$ deserves comment, because the raw images are far larger (up to several thousand pixels per side, varying by dataset) and DR grading depends on small lesions, above all the microaneurysms that define grade 1, which aggressive downsizing can attenuate. Two considerations justify it as the starting resolution. First, $224 \times 224$ is the input size at which both backbones were pretrained on ImageNet, so it makes transfer learning directly applicable (Section 2.4.3) and keeps the sixty training runs feasible on a single mid-range GPU; it is also the resolution at which comparable DR grading models operate, for example Kumar et al. (2025), who reach a QWK of 0.899 on APTOS 2019 at $224 \times 224$. Second, the five preprocessing techniques under study are themselves partly intended to make small lesions more salient before downsizing, so the resolution choice interacts with, rather than sits apart from, the factor being tested.

Because information loss from downsizing is nonetheless a genuine risk, two contingencies are planned in advance. If evaluation shows the minority classes, particularly grade 1 (mild), which hinges on microaneurysms, suffering disproportionately, the pipeline is designed to be re-run at a higher input resolution such as $384 \times 384$ or $512 \times 512$ (Plan B, at higher compute cost) and, failing that, with a tiling or patch-based scheme that preserves native-resolution detail within the region the retinal mask retains (Plan C). Any such change would be applied identically across all ten configurations so that the comparison stays fair, and would be reported explicitly.

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

The statistical analysis proceeds in three stages: an omnibus test, a descriptive ranking, and confirmatory pairwise tests on the leading configurations. As the omnibus stage, the ten configurations are compared across datasets with the Friedman test (Demsar, 2006), the standard non-parametric procedure for comparing several methods over multiple datasets. Within each dataset the ten configurations are ranked by QWK, and the Friedman test examines whether their average ranks differ significantly overall, as defined in Equation 3.1,

\begin{equation}
\chi_F^2 = \frac{12N}{k(k+1)}\left[\sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4}\right],
\label{eq:friedman}
\end{equation}

where $N=6$ datasets, $k=10$ configurations, and $R_j$ is the average rank of configuration $j$. QWK is the metric on which the test is run because it is the primary, ordinal-aware metric for DR grading; accuracy and macro-F1 are reported descriptively alongside it. When the Friedman test is significant, the average-rank ordering of the ten configurations is reported descriptively to show which configurations perform best across datasets. Because six datasets give the omnibus test limited power to resolve all ten configurations, this ranking is treated as descriptive evidence of consistency rather than as a set of pairwise significance claims, and the confirmatory testing below is restricted to the two leading configurations.

The two leading configurations (those with the best average rank across datasets) are then compared directly at two levels. Across datasets, they are compared with the Wilcoxon signed-rank test (Wilcoxon, 1945) on their six paired per-dataset QWK values, as defined in Equation 3.2,

\begin{equation}
W = \min(W^+, W^-), \qquad W^+ = \sum_{d_i > 0} \operatorname{rank}(|d_i|),
\label{eq:wilcoxon}
\end{equation}

where $d_i$ is the QWK difference between the two configurations on dataset $i$ and ranks are taken over $|d_i|$; the null hypothesis is that the median difference is zero. This is a single planned comparison of two methods, for which six paired datasets provide adequate power. Within each dataset, the same two configurations are compared with McNemar's test (McNemar, 1947) on the paired per-image predictions of that dataset's test partition, as defined in Equation 3.3,

\begin{equation}
\chi^2 = \frac{(b - c)^2}{b + c},
\label{eq:mcnemar}
\end{equation}

where $b$ and $c$ count the images that one configuration classifies correctly and the other does not; operating on thousands of paired samples, this test has substantially higher power within a single dataset. Because it is applied once per dataset, the family of six per-dataset McNemar tests is corrected with the Holm-Bonferroni procedure (Holm, 1979) to control the family-wise error rate.

Beyond comparing whole configurations, the factorial design also allows each factor to be tested separately, which maps directly onto RQ1 and keeps the number of comparisons small enough for six datasets to retain usable power. For the backbone factor, the per-dataset QWK scores are averaged over the five preprocessing techniques to give one value per backbone on each dataset, and ResNet-50 and ViT-B/16 are then compared across the six datasets with the Wilcoxon signed-rank test, a single planned comparison. For the preprocessing factor, the per-dataset QWK scores are averaged over the two backbones to give a $5 \times 6$ matrix, on which the five preprocessing techniques are compared with the Friedman test; when it is significant, each technique is compared against the baseline with pairwise Wilcoxon signed-rank tests under Holm-Bonferroni correction. The preprocessing $\times$ backbone interaction, that is, whether the best preprocessing technique differs between the two backbones, is examined descriptively by comparing the preprocessing ranking within each backbone separately.

The division of labour is explicit: the across-dataset Wilcoxon test answers whether the leading configuration is consistently better over the six datasets, whereas the within-dataset McNemar tests answer whether the two configurations differ on a given dataset. A configuration is regarded as robustly superior only when it both leads the cross-dataset ranking and wins the within-dataset McNemar comparisons. Following the principle of pre-specifying outcomes, if no configuration separates from the field, the conclusion that the best configuration is dataset-specific is reported as a substantive result consistent with the study's motivation (Section 1.1), not as an inconclusive finding.

The analysis is implemented with the \texttt{scipy.stats} and \texttt{statsmodels} libraries (Friedman and Wilcoxon signed-rank tests, McNemar's test, and the Holm-Bonferroni correction). In addition, Grad-CAM (Selvaraju et al., 2017) is generated on the best configuration for visual verification of the basis of the predictions.

### 3.8 Implementation Tools and Environment

The experimental pipeline is implemented in Python 3.11 using the following libraries: PyTorch 2.3 as the tensor and autodiff backend; torchvision 0.18 for ResNet-50 and the ImageNet-1k pretrained weights; timm 1.0 (Wightman, 2019) for ViT-B/16 and the ImageNet-21k pretrained weights; OpenCV 4.9 and scikit-image 0.22 for implementing the five preprocessing techniques; Albumentations 1.4 for the augmentation pipeline; scikit-learn 1.4 for computing the evaluation metrics; SciPy 1.13 and statsmodels 0.14 for the statistical tests (Friedman, Wilcoxon signed-rank, McNemar, and the Holm-Bonferroni correction); and Matplotlib 3.8 and seaborn 0.13 for visualising the confusion matrices and Grad-CAM heatmaps. The experiments are run in a single-GPU environment; a mid-range GPU with roughly 16 GB of memory is sufficient for the chosen batch size and input resolution, and the exact hardware may vary with the computational resources available at the time of training. All code, training scripts, evaluation scripts, configuration files, and metric logs are stored in a public Git repository accompanied by a reproduction README with explicit library versions, seeds, and execution commands.

### 3.9 Research Schedule

The research activities are planned to run for twelve months, from March 2026 to February 2027, covering literature study and proposal writing, the proposal seminar and its revision, environment preparation and collection of the six datasets, implementation of the preprocessing pipeline, training of the ten configurations on each of the six datasets, per-dataset evaluation, Friedman statistical analysis and Grad-CAM visualisation, the writing of Chapter IV and Chapter V, the results seminar, and the thesis defense and final submission. The monthly schedule is detailed in Table 3.4; the columns from March to December are months in 2026, while January and February are months in 2027. The schedule is indicative and may be adjusted according to the results of consultations with the supervisors and the availability of computational resources.

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

\refitem{Krause, J., Gulshan, V., Rahimy, E., Karth, P., Widner, K., Corrado, G. S., Peng, L., \& Webster, D. R. (2018). Grader Variability and the Importance of Reference Standards for Evaluating Machine Learning Models for Diabetic Retinopathy. \emph{Ophthalmology}, \emph{125}(8), 1264--1272. \texttt{https://doi.org/10.1016/j.ophtha.2018.01.034}}

\refitem{Kumar, S., Aditya, D. S., Kumar, T. L., Bikku, T., Thota, S., \& Kumar, C. (2025). Stage-Aware Diagnosis of Diabetic Retinopathy via Ordinal Regression. \emph{arXiv preprint} arXiv:2511.14398.}

\refitem{Li, T., Gao, Y., Wang, K., Guo, S., Liu, H., \& Kang, H. (2019). Diagnostic Assessment of Deep Learning Algorithms for Diabetic Retinopathy Screening. \emph{Information Sciences}, \emph{501}, 511--522.}

\refitem{Liu, R., Wang, X., Wu, Q., Dai, L., Fang, X., Yan, T., et al. (2022). DeepDRiD: Diabetic Retinopathy-Grading and Image Quality Estimation Challenge. \emph{Patterns}, \emph{3}(6), 100512.}

\refitem{Loshchilov, I., \& Hutter, F. (2019). Decoupled Weight Decay Regularization. \emph{International Conference on Learning Representations (ICLR)}.}

\refitem{McNemar, Q. (1947). Note on the Sampling Error of the Difference Between Correlated Proportions or Percentages. \emph{Psychometrika}, \emph{12}(2), 153--157.}

\refitem{Mok, D., Bum, J., Tai, L. D., \& Choo, H. (2024). Cross Feature Fusion of Fundus Image and Generated Lesion Map for Referable Diabetic Retinopathy Classification. \emph{arXiv preprint} arXiv:2411.03618.}

\refitem{Porwal, P., Pachade, S., Kokare, M., Deshmukh, G., Son, J., Bae, W., \ldots\ \& Meriaudeau, F. (2020). IDRiD: Diabetic Retinopathy Segmentation and Grading Challenge. \emph{Medical Image Analysis}, \emph{59}, 101561.}

\refitem{Saputra, N. A., Helvinda, W., \& Rahman, K. (2024). Prevalence and Risk Factors of Diabetic Retinopathy in a Tertiary Hospital in Padang, Indonesia. \emph{Bioscientia Medicina: Journal of Biomedicine and Translational Research}, \emph{9}(1), 219--231.}

\refitem{Sasongko, M. B., Widyaputri, F., Agni, A. N., Wardhana, F. S., Kotha, S., Gupta, P., Widayanti, T. W., Haryanto, S., Widyaningrum, R., Wong, T. Y., Kawasaki, R., \& Wang, J. J. (2025). Incidence and Progression of Diabetic Retinopathy and Blindness in Indonesian Adults with Type 2 Diabetes. \emph{PLoS ONE}, \emph{20}, e0322093.}

\refitem{Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., \& Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. \emph{Proceedings of the IEEE International Conference on Computer Vision (ICCV)}, 618--626.}

\refitem{Teo, Z. L., Tham, Y. C., Yu, M., Chee, M. L., Rim, T. H., Cheung, N., \ldots\ \& Cheng, C. Y. (2021). Global Prevalence of Diabetic Retinopathy and Projection of Burden through 2045: Systematic Review and Meta-Analysis. \emph{Ophthalmology}, \emph{128}(11), 1580--1591.}

\refitem{Tusfiqur, H. M., Nguyen, D. M. H., Truong, M. T. N., Nguyen, T. A., Nguyen, B. T., Barz, M., Profitlich, H.-J., Than, N. T. T., Le, N., Xie, P., \& Sonntag, D. (2022). DRG-Net: Interactive Joint Learning of Multi-lesion Segmentation and Classification for Diabetic Retinopathy Grading. \emph{arXiv preprint} arXiv:2212.14615.}

\refitem{Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., \& Polosukhin, I. (2017). Attention Is All You Need. \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 5998--6008.}

\refitem{Wightman, R. (2019). \emph{PyTorch Image Models (timm)} [Computer software]. GitHub. \texttt{https://github.com/huggingface/pytorch-image-models}}

\refitem{Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. \emph{Biometrics Bulletin}, \emph{1}(6), 80--83.}

\refitem{Wilkinson, C. P., Ferris III, F. L., Klein, R. E., Lee, P. P., Agardh, C. D., Davis, M., \ldots\ \& Verdaguer, J. T. (2003). Proposed International Clinical Diabetic Retinopathy and Diabetic Macular Edema Disease Severity Scales. \emph{Ophthalmology}, \emph{110}(9), 1677--1682.}

\refitem{Wong, T. Y., \& Sabanayagam, C. (2023). The War on Diabetic Retinopathy: Where Are We Now? \emph{Asia-Pacific Journal of Ophthalmology}, \emph{12}(3), 213--221.}

\refitem{Zhou, Y., Chia, M. A., Wagner, S. K., Ayhan, M. S., Williamson, D. J., Struyven, R. R., \ldots\ \& Keane, P. A. (2023). A Foundation Model for Generalizable Disease Detection from Retinal Images. \emph{Nature}, \emph{622}(7981), 156--163.}

\refitem{Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization. In P. S. Heckbert (Ed.), \emph{Graphics Gems IV} (pp. 474--485). Academic Press.}
