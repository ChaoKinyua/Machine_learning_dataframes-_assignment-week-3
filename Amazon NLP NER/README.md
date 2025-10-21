# 🛍️ Amazon Product Review NLP Analysis

A Natural Language Processing project that analyzes Amazon product reviews to extract named entities (products and brands) and perform sentiment analysis using spaCy and rule-based approaches.

## 🎯 Project Overview

This prototype demonstrates the application of NLP techniques to understand customer sentiment and automatically extract product information from unstructured review text.

**Key Features:**
- Named Entity Recognition (NER) for products and brands
- Rule-based sentiment analysis with polarity scoring
- Brand sentiment rankings and comparative analysis
- Interactive visualizations and exportable insights

## 🛠️ Technologies Used

- **spaCy** (`en_core_web_sm`) - Named Entity Recognition
- **spaCyTextBlob** - Sentiment analysis
- **Pandas** - Data manipulation and analysis
- **Plotly & Matplotlib** - Data visualization
- **Python 3.x** - Core programming language

## 📊 Key Results

Based on analysis of Amazon product reviews:
- Successfully extracted products and brand mentions from review text
- Achieved sentiment classification with correlation to star ratings
- Generated brand performance rankings
- Created actionable insights through visualizations

## 🚀 How to Run

### Google Colab (Recommended for Quick Demo)

1. Open the notebook in Google Colab
2. Run **Runtime → Run all**
3. Upload your Amazon reviews CSV when prompted
4. View results in 3-5 minutes

### Local Setup

```bash
# Clone repository
git clone <repository-url>
cd amazon-nlp-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run analysis
python main.py
```

## 📁 Project Structure

```
amazon-nlp-analysis/
├── notebooks/
│   └── Amazon_NLP_Analysis.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── ner_extractor.py
│   ├── sentiment_analyzer.py
│   └── visualizations.py
├── outputs/
│   ├── results.csv
│   └── analysis_summary.txt
├── requirements.txt
└── README.md
```

## 📈 Sample Output

```
Review: "The Samsung Galaxy has an amazing camera but terrible battery life."

Extracted Entities:
  • Products: Samsung Galaxy
  • Brands: Samsung

Sentiment Analysis:
  • Overall: Mixed (Score: 0.15)
  • Classification: Neutral
```

## 🔮 Future Enhancements

- [ ] Deploy as interactive web application using Streamlit
- [ ] Train custom NER model on Amazon-specific product entities
- [ ] Implement aspect-based sentiment analysis
- [ ] Add multi-language support
- [ ] Scale to process millions of reviews
- [ ] Create REST API for real-time analysis

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Collaborators

- **[Maliph-Guye]** - *Developer* - [GitHub Profile](https://github.com/Maliph-Guye) | [LinkedIn](https://linkedin.com/in/tonnis-ondito-354077224)

### Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](link-to-issues).

To contribute:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**[Your Name]**
- Email: guyemaliph@gmail.com
- GitHub: [Maliph-Guye](https://github.com/Maliph-Guye)
- LinkedIn: [LinkedIn](https://linkedin.com/in/tonnis-ondito-354077224)
- Portfolio: [View Portfolio](https://maliph-guye.github.io/My-Blog-Page/index.html)

---

## 🙏 Acknowledgments

- Dataset: [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets) from Kaggle
- spaCy documentation and community
- Natural Language Processing course materials and tutorials

---

<div align="center">
  
**⭐ Star this repository if you found it helpful!**


</div>