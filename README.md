# Hybrid Drift Score Metric (HDSM)

## Project Overview

The Hybrid Drift Score Metric (HDSM) is a domain-independent data drift detection framework developed as part of an M.Sc. Data Science project.

The system monitors incoming data windows and detects changes in data distributions that may impact machine learning model performance. HDSM can be applied across multiple domains including healthcare, finance, IoT, and business analytics.

---

## Features

* Real-time drift monitoring
* Window-based analysis
* Hybrid Drift Score Metric (HDSM)
* Interactive Streamlit dashboard
* CSV-based streaming simulation
* Drift visualization and reporting

---

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn

---

## Project Structure

```text
HDSM_Project/
│
├── apps/
│   └── app.py
│
├── Data/
├── notebooks/
├── plots/
├── results/
│
├── diabetes.csv
├── hdsm.ipynb
├── hdsm_results.csv
└── README.md
```

---

## Dataset Support

The HDSM framework is domain-independent and can analyze any structured tabular dataset.

Example application domains include:

* Healthcare Analytics
* Financial Risk Analytics
* IoT Sensor Monitoring
* Business Intelligence
* Customer Analytics
* Manufacturing Data Streams

The diabetes dataset included in this repository is used only as a demonstration dataset for validating drift detection capabilities.

---

## How to Run

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Start the Streamlit application

```bash
streamlit run apps/app.py
```

3. Open the URL displayed in the terminal.

---

## Results

The system generates:

* Drift scores
* Drift classification
* Visualization plots
* Drift monitoring reports

---

## Future Enhancements

* Apache Kafka integration for real-time streaming
* AWS cloud deployment
* Automated alerting using AWS SNS
* Storage of drift logs in AWS S3
* Automated model retraining using AWS SageMaker
* Production-scale monitoring architecture

---

## Author

Riddhi Hajare

M.Sc. Data Science

University of Mumbai
