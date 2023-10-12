# oneAPI-CodeMaven_PrototypePhase
[Intel® Devmesh Project](https://devmesh.intel.com/projects/efficient-ecg-auto-diagnosis-application-for-underserved-communities-using-intel-oneapi-ai-analytics-toolkit-in-intel-devcloud-jupyterhub)

#### Theme : 

Healthcare for underserved communities 

#### Problem Statement : 

Building an Efficient [ECG Auto-diagnosis Application](https://huggingface.co/spaces/deepthiaj/Electro_oneAPI) for Underserved Communities using Intel® oneAPI AI Analytics Toolkit in Intel® DevCloud JupyterHub

![image](https://user-images.githubusercontent.com/70970105/236674030-99e2aba8-2711-4bbc-80d8-6e3285a3fbcd.png)

[How can we use technology to improve access to healthcare for diagnosis of heart diseases amongst the underserved communities?](https://medium.com/@deepthiaj/building-an-efficient-ecg-auto-diagnosis-application-for-underserved-communities-using-intel-a05059829739)

The theme is focused on leveraging technology to create solutions that can be easily scalable, affordable, and accessible to everyone, regardless of their background or circumstances.

More detailed [experience through the development of this project can be found in this article](https://medium.com/@deepthiaj/my-experience-using-the-intel-oneapi-ai-toolkit-in-developing-an-efficient-ecg-application-for-9679763aa167). The [web application using Streamlit is build upon the performance benefits offered by Intel oneAPI toolkit](https://medium.com/@deepthiaj/streamlit-and-intel-oneapi-support-unleashing-the-power-of-interactive-data-visualization-and-61bbabe7f417). The [XGBoost and daal4py models used in the application also leverages the Intel oneAPI](https://medium.com/@deepthiaj/xgboost-vs-daal4py-unleashing-the-power-of-intel-oneapi-in-machine-learning-96b1836051f4) benefits. The training and evaluation was primarily done using [Intel oneAPI integration within JupyterHub environment](https://medium.com/@deepthiaj/empowering-machine-learning-unleashing-the-benefits-of-oneapi-tools-9fef3cd972da) using the reliable [15-lead ECG dataset with diagnosis information provided as the PTB ECG diagnostic database](https://medium.com/@deepthiaj/working-with-large-datasets-2228a2a1e25f).

#### About Intel® DevCloud JupyterHub

JupyterHub is the next-generation user interface for Intel® DevCloud, which includes existing Jupyter Notebooks and a simplified modular modular structure, where user can open several notebooks or files (e.g. HTML, Text, Markdowns etc) as tabs in the same window.

JupyterLab offers much more of a “modern” IDE-like experience to developers:

~ High performance experience compared to singular inline browser experience . See image : can be customized for remote/distance learning across the world
~ Public/Private APIs are well separated
~ Enables JavaScript development for npm-based development/packaging (supports Typescript & Phosphor.js)
~ User can add 3rd party extensions

![image](https://user-images.githubusercontent.com/70970105/236620109-6958d842-012b-4a0a-9c94-3e8e2120f202.png)

JupyterLab is a popular web-based interactive development environment for data science, machine learning, and scientific computing. It provides a flexible and extensible environment where you can create, edit, and run code cells, visualize data, and generate rich documents that combine code, visualizations, and explanatory text.

[OneAPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html#gs.1dwtst) is an open, standards-based unified programming model from Intel that aims to simplify development across diverse computing architectures, including CPUs, GPUs, FPGAs, and other accelerators. It allows developers to write high-performance code that can seamlessly target different hardware platforms without the need for extensive modifications.

JupyterLab with oneAPI integration combines the power of JupyterLab's interactive and collaborative environment with the capabilities of OneAPI's hardware-agnostic programming model. This integration allows developers and data scientists to leverage OneAPI's libraries and tools directly within JupyterLab to develop and optimize applications for various hardware architectures.

By using JupyterLab with OneAPI, you can write and execute code that takes advantage of the performance and parallelism offered by different hardware devices, such as GPUs and FPGAs. It provides a unified interface for writing and running code across these different architectures, making it easier to explore and optimize your algorithms for specific hardware targets.

With JupyterLab and OneAPI, you can prototype, debug, and optimize your code interactively, visualize data and performance metrics, and document your work in a reproducible and shareable manner. It empowers you to explore the full potential of heterogeneous computing and accelerate your development and research workflows.

The Intel oneAPI AI Analytics Toolkit (AI Kit) provides a consolidated package of Intel’s latest deep and machine learning optimizations all in one place with seamless interoperability and high performance. The AI Kit includes Intel-optimized versions of deep learning frameworks, Python libraries, and a lightweight parallel data frame to streamline end-to-end data science and AI workflows on Intel architectures. The AI Kit includes the Intel Distribution for Python (including optimizations for XGBoost, daal4py, and more),.

Using Intel® Optimized Frameworks and Intel® oneAPI AI Analytics Toolkit and Libraries to create solutions that can be focused on addressing any aspect of healthcare access, such as financial barriers, lack of transportation, limited healthcare resources, language barriers etc. The related libraries and optimizations for improving performance which will make the solution stand-out in terms of performance are: oneAPIDeep Neural Network Library, Intel oneAPI Math Kernel Library (oneMKL), Intel® oneAPI Threading Building Blocks, Intel® oneAPI Data Analytics Library, Intel® oneAPI DPC++ Library, Intel® Optimization for TensorFlow, Intel® Distribution for Python, Intel® Extension for Scikit-learn etc.

#### About PTB Diagnostic ECG Database

![image](https://user-images.githubusercontent.com/70970105/236620071-ed070db9-4626-4fd5-9832-3a6b26ce0e0c.png)



[PTB Diagnostic ECG Database](https://www.physionet.org/content/ptbdb/1.0.0/), is a collection of 549 high-resolution 15-lead ECGs (12 standard leads together with Frank XYZ leads), including clinical summaries for each record. From one to five ECG records are available for each of the 294 subjects, who include healthy subjects as well as patients with a variety of heart diseases.

###### Data Description

The ECGs in this collection were obtained using a non-commercial, PTB prototype recorder with the following specifications:

16 input channels, (14 for ECGs, 1 for respiration, 1 for line voltage)
Input voltage: ±16 mV, compensated offset voltage up to ± 300 mV
Input resistance: 100 Ω (DC)
Resolution: 16 bit with 0.5 μV/LSB (2000 A/D units per mV)
Bandwidth: 0 - 1 kHz (synchronous sampling of all channels)
Noise voltage: max. 10 μV (pp), respectively 3 μV (RMS) with input short circuit
Online recording of skin resistance
Noise level recording during signal collection

The database contains 549 records from 290 subjects (aged 17 to 87, mean 57.2; 209 men, mean age 55.5, and 81 women, mean age 61.6; ages were not recorded for 1 female and 14 male subjects). Each subject is represented by one to five records. There are no subjects numbered 124, 132, 134, or 161. Each record includes 15 simultaneously measured signals: the conventional 12 leads (i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6) together with the 3 Frank lead ECGs (vx, vy, vz). Each signal is digitized at 1000 samples per second, with 16 bit resolution over a range of ± 16.384 mV. On special request to the contributors of the database, recordings may be available at sampling rates up to 10 KHz.

Within the header (.hea) file of most of these ECG records is a detailed clinical summary, including age, gender, diagnosis, and where applicable, data on medical history, medication and interventions, coronary artery pathology, ventriculography, echocardiography, and hemodynamics. The clinical summary is not available for 22 subjects. The diagnostic classes of the remaining 268 subjects are summarized below:

Diagnostic class	(Number of subjects)
Myocardial infarction	(148)
Cardiomyopathy/Heart failure	(18)
Bundle branch block	(15)
Dysrhythmia	(14)
Myocardial hypertrophy	(7)
Valvular heart disease	(6)
Myocarditis	(4)
Miscellaneous	(4)
Healthy controls	(52)

###### References

Bousseljot, R.; Kreiseler, D.; Schnabel, A. Nutzung der EKG-Signaldatenbank CARDIODAT der PTB über das Internet. Biomedizinische Technik, Band 40, Ergänzungsband 1 (1995) S 317

Kreiseler, D.; Bousseljot, R. Automatisierte EKG-Auswertung mit Hilfe der EKG-Signaldatenbank CARDIODAT der PTB. Biomedizinische Technik, Band 40, Ergänzungsband 1 (1995) S 319


