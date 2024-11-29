<div align="center">

# 🔍 Peer Review Analyzer

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io)
[![OpenReview](https://img.shields.io/badge/OpenReview-API%20v2-green.svg)](https://openreview.net)

A comprehensive tool for analyzing peer reviews from OpenReview.net, featuring data scraping, analysis, and interactive visualization capabilities.

[Installation](#installation) •
[Features](#features) •
[Usage](#usage) •
[Documentation](#documentation)

</div>

---

## Overview

This project provides tools to:
1. Scrape peer review data from OpenReview.net
2. Analyze review patterns, sentiments, and trends
3. Visualize review data through an interactive Streamlit dashboard

## Installation

```bash
# Clone the repository
git clone https://github.com/Ahanmr/peer-review-analyzer.git
cd peer-review-analyzer

# Install dependencies
pip install -r requirements.txt
```

## Features

### 1. Data Scraping
- Extract reviews from multiple conferences (NeurIPS, ICLR, etc.)
- Support for both OpenReview API v2 and direct HTTP requests
- Automatic rate limiting and error handling
- Compressed CSV export functionality

### 2. Analysis Capabilities
- Sentiment analysis of review text
- Statistical analysis of ratings and confidence scores
- Temporal patterns in review submissions
- Reviewer behavior analysis
- Keyword extraction and topic modeling

### 3. Interactive Visualization
- Real-time data exploration
- Multiple visualization types
- Customizable filters and parameters
- Export capabilities for further analysis

## Usage

### Running the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard includes:

1. **Overview Page**
   - Total reviews and papers statistics
   - Score distributions
   - Confidence metrics
   - Quick insights dashboard

2. **Embeddings Visualization**
   - t-SNE and UMAP visualizations
   - Interactive clustering analysis
   - Customizable color coding by metrics

3. **Review Analysis**
   - Sentiment distribution plots
   - Rating vs. Sentiment correlation
   - Temporal pattern analysis
   - Statistical summaries

4. **Word Clouds**
   - Separate visualizations for positive/negative reviews
   - Custom stopword filtering
   - Interactive word frequency analysis

5. **Reviewer Insights**
   - Score correlation matrices
   - Reviewer behavior patterns
   - Time-based analysis tools

6. **Keyword Search**
   - Context-aware search functionality
   - Relevance scoring
   - Highlighted results display

### Running the Scraper

```bash
python scraper.py
```

The scraper supports:
- Conference selection
- Year range specification
- Custom data filtering
- Rate-limited API requests
- Error handling and retry logic

### Running the Analysis Tool

```bash
python analysis.py
```

Features include:
- Automated review analysis
- CSV export functionality
- Statistical computations
- Visualization generation

## Documentation

### OpenReview Data Access

#### 1. Using OpenReview-py Library (API v2)

```python
from openreview import api

# Initialize client
client = openreview.api.OpenReviewClient(
    baseurl='https://api2.openreview.net',
    username='your_username',
    password='your_password'
)

# Access conference data
conference = client.get_group('conference_id')
submissions = client.get_all_notes(invitation='conference_id/-/Submission')
```

#### 2. Using Direct API Requests

```python
import requests

# Example API endpoint
base_url = 'https://api.openreview.net/notes'
params = {
    'invitation': 'conference.cc/year/Conference/-/Blind_Submission',
    'offset': 0,
    'limit': 100
}
response = requests.get(base_url, params=params)
```

### Configuration

Create a `.env` file in the project root:

```env
OPENREVIEW_USERNAME=your_username
OPENREVIEW_PASSWORD=your_password
```

### Data Format

Reviews are stored in compressed CSV format with the following columns:
- `paper_id`: Unique identifier for the paper
- `review_id`: Unique identifier for the review
- `confidence`: Reviewer's confidence score
- `rating`: Review rating
- `review_text`: The actual review content
- `timestamp`: When the review was submitted

## Acknowledgments

- OpenReview.net for providing access to peer review data
- Streamlit for the interactive visualization framework
- NLTK and spaCy for NLP capabilities

---

<div align="center">
Made for the research community
</div>
