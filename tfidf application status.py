# create an automatic status cleanser, taking in a raw decision and status pair and returning an automatically cleaned version and confidence score, or else flagging as "other", to be checked by hand, and added back into the training set, improving success over time

#note that i wouldnt run this on a large body of previous applications as it will take too long. Rather, do a basic join first, to catch those where the exact pairing has been seen before. that should knock off most examples, and you can auto tag the remaining. 

#this kind of function is most useful in BAU, where it can run on new applications as they come through, automatically tagging most, and flagging up previously unseen statuses for review, which in itself is useful information to monitor the system by


#%% 
# import packages

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


#read in the training set that i coded up myself - not yet checked by anyone who knows what they are doing!
textStatus = pd.read_csv("C:/Users/tomsh/OneDrive/Desktop/PlanneR/Housing Land/StatusCleaned.csv")

# prep the training data
# here we are creating a combined decision and status, which helps to flag up those where the two appear to contradict each other (decision says approve, status says reject), and makes sure they get logged for checking by a human
textStatus = textStatus.fillna(" ")
textStatus["decStat"] = textStatus["stat_desc"] + " " +  textStatus["decis_desc"]


# Download NLTK resources (if not already installed)
nltk.download('punkt')
nltk.download('stopwords')

#pre-process the training text

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())  # Lowercasing
    # Removing stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    #bespoke stopwords - we can add to this as we identify more words, this is more of a placeholder for now
    tokens = [t for t in tokens if t not in ["planning","permission","application"]]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]
    return ' '.join(tokens)

# Apply the preprocessing to each proposal in the training
textStatus['processed_decStat'] = textStatus["decStat"].apply(preprocess_text)

# Set up a TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(textStatus['processed_decStat'])

# Convert the TF-IDF matrix to a DataFrame:
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=textStatus.index)

#Define your auto-tagger


def interpretStatus(new_proposal_text):
    # Assuming new_proposal_text is a string containing your new proposal text
    
    # Preprocess the new proposal
    processed_new_proposal = preprocess_text(new_proposal_text)

    # Transform the new proposal into a TF-IDF vector
    new_tfidf_vector = vectorizer.transform([processed_new_proposal])

    # Compute cosine similarity
    # Note: this will give you a similarity score for each proposal in your existing dataset
    cosine_similarities = cosine_similarity(new_tfidf_vector, tfidf_matrix).flatten()

    # Get the index of the most similar proposal(s) - in this case the top 10 most similar proposals
    top_similar_indices = np.argsort(cosine_similarities)[-10:]

    # we can then check what the top ten most similar Cleaned Statuses were in the trainign set
    most_similar_target = pd.Series(textStatus.iloc[top_similar_indices]['StatusCleaned'].values)
    most_similar_scores = pd.Series(cosine_similarities[top_similar_indices])
    
    # filter to only the similar statuses with a high score, and return the modal suggestion, along with a confidence score, else return "Other" 

    suggestionDF = pd.DataFrame({"Status":most_similar_target,"Score": most_similar_scores})
    #filter suggestions that arent at least X match
    suggestionDF = suggestionDF[suggestionDF["Score"]>0.75]
    if len(suggestionDF) > 0:
        suggestion = suggestionDF["Status"].mode()[0]
        certainty = sum(suggestionDF["Status"] == suggestion) / len(suggestionDF["Status"])
        return suggestion, certainty
    else:
        return "Other", -99


# test it out 
#this one should clearly be coded as approved
interpretStatus("Approved We approved this one")

#this one is ambiguous - probably best to flag it as other and have someone check by eye
interpretStatus("insufficient fee approved")


# %% apply in practice

appsWithUnmatchedStatus = "replace with a filtered version of your Cloud Connector dataframe, for those where the exact status/decision pairing isn't in the training set"

appsWithUnmatchedStatus["decStat"] = appsWithUnmatchedStatus["stat_desc"].fillna("") + " " +  appsWithUnmatchedStatus["decis_desc"].fillna("")

appsWithUnmatchedStatus[['statusMachine', 'PredictionScore']] = appsWithUnmatchedStatus["decStat"].apply(lambda x: pd.Series(interpretStatus(x)))



