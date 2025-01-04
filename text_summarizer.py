from transformers import pipeline

# Load a summarization pipeline using a pre-trained model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Your text to summarize
text_to_summarize = """
MSCI’s Asia Pacific Index - a gauge for benchmarks in the region - slipped the most in three months as benchmarks from Hong Kong to Japan, South Korea were all in the red. Futures contracts for US shares steadied in Asian trading after the S&P 500 erased earlier gains and fell more than 1% in a volatile session.

China unexpectedly weakened its yuan defense as pressure from a resurgent dollar and poor sentiment pressured it toward a policy red line. A gauge of emerging-markets currencies fell to a year-to-date low after the offshore yuan slid in reaction to the weaker reference rate.

Investors will now turn their focus to China’s torrent of economic indicators Tuesday. The slowdown in the nation’s economy in the first quarter probably wouldn’t be a strong start to a year in which the government has set an ambitious 5% target, according to Bloomberg Economics. Ten-year bond yields are poised to break to the lowest level since 2002 on growth expectations.
"""

# Perform summarization
summary = summarizer(text_to_summarize, max_length=130, min_length=30, do_sample=False)

# Print summarized text
print(summary[0]['summary_text'])

