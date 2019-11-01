# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:16:37 2018

@author: Stephen

Cluster Script
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 21:08:52 2018

@author: Stephen

Conference Call Transcript Analysis Module
"""

"""
Manager and Analyst Text Work -- w/ Jason Kotter

XML Call Transcript Navigation Guide:
    Event:
        eventTypeName
        eventTypeId
        lastUpdate
        Id
    EventStory:
        Id
        version
        storyType
        action
        expirationDate
    Headline
    Body
    eventTitle
    city
    companyName
    companyTicker
    startDate
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from collections import defaultdict
import simplejson
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
LdaModel = gensim.models.ldamodel.LdaModel
import scipy.spatial.distance as sd
import os
import time
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter

# must do nltk.download(), change file path to the scratch folder, then run below command
nltk.data.path.append('/storage/home/sro5/scratch/nltk_data')
# load in McDonald sentiment words list
words = pd.read_csv('/storage/home/sro5/Desktop/pythoncode/conference_call_data/words_df.csv')
# Separate words data into respective sentiment lists
pos_list = list(pd.Series(np.where(words['Positive']==1, words['Word'], np.nan)).dropna())
neg_list = list(pd.Series(np.where(words['Negative']==1, words['Word'], np.nan)).dropna())
uncert_list = list(pd.Series(np.where(words['Uncertainty']==1, words['Word'], np.nan)).dropna())
litig_list = list(pd.Series(np.where(words['Litigious']==1, words['Word'], np.nan)).dropna())
constr_list = list(pd.Series(np.where(words['Constraining']==1, words['Word'], np.nan)).dropna())

def read_file(filepath):
    """ Facilitates reading in xml file, and reading multiple xml files simultaneously.
    """
    
    with open(filepath) as fp:
        soup = BeautifulSoup(fp, 'xml')
    fp.close()
    
    return soup


def get_attributes(soup_xml_file):
    """ Function to compile the attributes of each conference call into a structured
    row DataFrame. This will allow each call's attributes to easily concat for 
    a complete DataFrame of the call data details.
    
    Additionally, the function returns the text body of the call transcript for
    further analysis. 
    """
    
    date = pd.to_datetime(str(soup_xml_file.startDate.text))
    name = str(soup_xml_file.companyName.text)
    ticker = str(soup_xml_file.companyTicker.text)
    fileid = int(str(soup_xml_file.Event['Id']))
    calltype = str(soup_xml_file.Event['eventTypeName'])
    body = soup_xml_file.Body.text
    
    df = pd.DataFrame({'date':date, 'name':name, 'ticker':ticker,
                       'file_id':fileid, 'call_type':[calltype]})

    return body, df


def subdivide(lines, regex_pattern):
    """ Divides the call by specified regex pattern.
    
    Example:
        Divides the body of the call into 6 sections:
            0. Title/Header
            1. "Corporate Participants" string
            2. Names of corporate participants
            3. "Conference Call Paricipants" string
            4. Names of conference call participants
            5. Body of the call (call discussion)
    """
    
    equ_pattern = re.compile(regex_pattern, re.MULTILINE)
    sections = equ_pattern.split(lines)
    sections = [section.strip('\n') for section in sections]
    return sections


def process_dashed_sections(section):
    """ Creates a dictionary with empty values where each key is a section of the 
    body of the call. 
    """
    
    subsections = subdivide(section, "^-+")
    heading = subsections[0]  # header of the section - either "Presentation" or "Q&A"
    d = {key.strip(): value.strip() for key, value in zip(subsections[1::2], subsections[2::2])}
    return heading, d


def assign_attendee(d, attendees):
    """ Fills dictionary with speakers role, job, and respective text (empty). """
    
    new_d = defaultdict(list)
    for key, value in d.items():
        a = [a for a in attendees if a in key]
        if len(a) == 1:
            # to strip out any additional whitespace anywhere in the text including '\n'.
            new_d[a[0]].append(value.strip())
        elif len(a) == 0:
            # to strip out any additional whitespace anywhere in the text including '\n'.
            new_d[key] = value.strip()
    # Add in job and participant type info
    new_d2 = defaultdict(defaultdict)
    for key, value in new_d.items():
        a = [a for a in attendees if a in key]
        if len(a) == 1:
            new_d2[a[0]]['job'] = attendees[a[0]][0]
            new_d2[a[0]]['group'] = attendees[a[0]][1]
            new_d2[a[0]]['text'] = value
        # If person doesn't appear in list of participants, we don't have info on job/group
        elif len(a) == 0:
            new_d2[key]['job'] = None
            new_d2[key]['group'] = None
            new_d2[key]['text'] = value
    return new_d2


def assign_text_to_speaker(body):
    """ Fills values of dictionary with speakers role, job, and respective text. 
    The full call transcript is populated throughout the values of the dictionary.
    """
    
    sections = subdivide(body, "^=+")
    # regex pattern for matching headers of each section
    header_pattern = re.compile("^.*[^\n]", re.MULTILINE)

    # regex pattern for matching the sections that contains
    # the list of attendee's (those that start with asterisks )
    if unicode("Corporate Participants", "utf-8") in sections:
        ppl_pattern = re.compile("^(\s+\*)(.+)(\s.*)", re.MULTILINE)
    else:
        ppl_pattern = re.compile("^(\s+\*)(\s.*)", re.MULTILINE)
        
    # regex pattern for matching sections with subsections in them.
    dash_pattern = re.compile("^-+", re.MULTILINE)

    ppl_d = dict()
    ppl_d['Operator'] = ['Operator', 'Operator']
    talks_d = dict()

    header = []
    # Step2. Handle each section like a switch case
    for section in sections:
        # Handle headers
        if len(section.split('\n')) == 1:  # likely to match only a header (assuming )
            header = header_pattern.match(section).string
    
        # Handle attendees/presenters
        elif ppl_pattern.match(section):
            if unicode("Corporate Participants", "utf-8") in sections:
                ppls = ppl_pattern.findall(section)
                d = {key.strip(): value.strip() for (_, key, value) in ppls}
            else:
                ppls = ppl_pattern.findall(section)
                ppls_list = []
                for i in ppls:
                    val = unicode('particiapnt', 'utf-8')
                    ppls_new = i + (val,)
                    ppls_list.append(ppls_new)
                d = {key.strip(): value.strip() for (_, key, value) in ppls_list}
            
            # assuming that if the previous section was detected as a header, then this section will relate
            # to that header
            if header:
                for key, value in d.items():
                    d[key] = [value, header]
            ppl_d.update(d)
    
        # Handle Presentations/Q&A subsections
        elif dash_pattern.findall(section):
            heading, d = process_dashed_sections(section)
            talks_d.update({heading: d})                  
            
        # Else its just some random text.
        else:
    
            # assuming that if the previous section was detected as a header, then this section will relate
            # to that header
            if header:
                talks_d.update({header: section})            

    # To assign the talks material (as a list) to the appropriate attendee/presenter. Still works if no match found.
    for key, value in talks_d.items():
        talks_d[key] = assign_attendee(value, ppl_d)
    
    return talks_d


def divide_in_two(body, sections):
    """ Splits call transcript between the prepared remarks and question and answer
    section. Returns the outputs as separte variables. 
    """
    if 'Questions and Answers' in sections[-1][:50]:
        prepared = sections[-2]
        qa = sections[-1]
    else:
        split_location = body.find('first question')
        prepared = body[:split_location]
        qa = body[split_location:]
    
    return prepared, qa


def prepared_qa_split(body):
    """ Fills values of dictionary with speakers role, job, and respective text. 
    The full call transcript is populated throughout the values of the dictionary.
    """
    
    sections = subdivide(body, "^=+")
    pre, qa = divide_in_two(body, sections)
    
    del sections[-1]
    
    sections.extend([pre, qa])
    
    # regex pattern for matching headers of each section
    header_pattern = re.compile("^.*[^\n]", re.MULTILINE)

    # regex pattern for matching the sections that contains
    # the list of attendee's (those that start with asterisks )
    if unicode("Corporate Participants", "utf-8") in sections:
        ppl_pattern = re.compile("^(\s+\*)(.+)(\s.*)", re.MULTILINE)
    else:
        ppl_pattern = re.compile("^(\s+\*)(\s.*)", re.MULTILINE)

    # regex pattern for matching sections with subsections in them.
    dash_pattern = re.compile("^-+", re.MULTILINE)

    ppl_d = dict()
    ppl_d['Operator'] = ['Operator', 'Operator']
    talks_d = dict()

    header = []
    # Step2. Handle each section like a switch case
    for section in sections:
    
        # Handle headers
        if len(section.split('\n')) == 1:  # likely to match only a header (assuming )
            header = header_pattern.match(section).string
    
        # Handle attendees/presenters
        elif ppl_pattern.match(section):
            if unicode("Corporate Participants", "utf-8") in sections:
                ppls = ppl_pattern.findall(section)
                d = {key.strip(): value.strip() for (_, key, value) in ppls}
            else:
                ppls = ppl_pattern.findall(section)
                ppls_list = []
                for i in ppls:
                    val = unicode('particiapnt', 'utf-8')
                    ppls_new = i + (val,)
                    ppls_list.append(ppls_new)
                d = {key.strip(): value.strip() for (_, key, value) in ppls_list}
            
            # assuming that if the previous section was detected as a header, then this section will relate
            # to that header
            if header:
                for key, value in d.items():
                    d[key] = [value, header]
            ppl_d.update(d)
    
        # Handle Presentations/Q&A subsections
        elif dash_pattern.findall(section):
            heading, d = process_dashed_sections(section)
            talks_d.update({heading: d})                  
            
        # Else its just some random text.
        else:
    
            # assuming that if the previous section was detected as a header, then this section will relate
            # to that header
            if header:
                talks_d.update({header: section})            

    # To assign the talks material (as a list) to the appropriate attendee/presenter. Still works if no match found.
    for key, value in talks_d.items():
        talks_d[key] = assign_attendee(value, ppl_d)
    # Rename keys to corresponding section of call (Prepared or Q&A)
    for k,i in zip(sorted(talks_d), range(2)):
        if i == 0:
            talks_d['prepared'] = talks_d.pop(k)
        elif i == 1:
            talks_d['qa'] = talks_d.pop(k)
    
    return talks_d


def group_assignment(call_dictionary, split_dictionary):
    """ Function that assigns call participant to be either a manager or analyst
    when the call transcript does not specify. Returns a dictionary of the 
    entire call that can then be processed and split between managers and 
    analysts.
    
    Must process data through the "assign_text_to_speaker" and "prepared_qa_split"
    functions before running through group_assignment.
    """
    # Rewrite to modify the split dictionary
    corp_list = []
    for i in split_dictionary['prepared'].keys():
        for k,v in call_dictionary.items():
            for p in v.keys():
                if i == p:
                    call_dictionary[k][p]['group'] = 'Corporate Participants'
                    corp_list.append(i)
                    
    conf_list = []
    for k,v in call_dictionary.items():
        for p in v.keys():
            conf_list.append(p)
    
    [conf_list.remove(x) for x in corp_list]
    
    for i in conf_list:
        for k,v in call_dictionary.items():
            for p in v.keys():
                if i == p:
                    call_dictionary[k][p]['group'] = 'Conference Call Participants'
    # Create new master dictionary of entire call with correct group assignments
    for k,v in call_dictionary.items():
        for p in v.keys():
            if p == 'Operator' or p == 'Moderator':
                call_dictionary[k][p]['group'] = 'Conference Call Participants'

    # Create two new dictionaries for the new split dictionary with correct group assignments
    #manager_dictionary = {}
    #analyst_dictionary = {}
    #for k,v in call_dictionary.items():
    #   for p,q in v.items():
    #        if q['group'] == 'Corporate Participants':
    #            manager_dictionary[p] = q
    #        elif q['group'] == 'Conference Call Participants':
    #            analyst_dictionary[p] = q

    return call_dictionary #, manager_dictionary, analyst_dictionary


def group_managers_analysts(call_dictionary, split_dictionary):
    """ Function that returns a correctly specified/named split dictionary between
    the prepared remarks and Q&A section. Gives each participant the correct
    group name in the correct section of the call.
    
    The input "call_dictionary" must be a dictionary that first passed through
    the "assign_to_text_speaker" function.
    """
    # Update split dictionary q&a with correct group assignments
    manager_list = []
    analyst_list = []
    for k,v in call_dictionary.items():
        for i,j in v.items():
            if j['group'] == 'Corporate Participants':
                manager_list.append(i)
            elif j['group'] == 'Conference Call Participants':
                analyst_list.append(i)
            
    for k,v in split_dictionary['prepared'].items():
        if k in manager_list:
            v['group'] = 'Corporate Participants'
            
    for k,v in split_dictionary['qa'].items():
        if k in manager_list:
            v['group'] = 'Corporate Participants'
        elif k in analyst_list:
            v['group'] = 'Conference Call Participants'

    return split_dictionary

def qa_group_managers_analysts(new_split_dictionary):
    """ Function that splits the q&a section between managers and analysts. """
    
    # Return the split q&a section of the calls
    manager_dict = {}
    analyst_dict = {}
    # Iterate through each person within the dictionary, save their information
    # to their respective new dictionary
    # Group only the entire call
    for k,v in new_split_dictionary['qa'].items():
        if v['group'] == 'Corporate Participants':
            manager_dict[k] = v
        elif v['group'] == 'Conference Call Participants':
            analyst_dict[k] = v
        
    return manager_dict, analyst_dict


def get_manager_text(new_split_dictionary):
    """ Function to collect all manager text into one dictionary. """
    
    manager_dict = {}
    for k,v in new_split_dictionary.items():
        for i,j in v.items():
            if j['group'] == 'Corporate Participants':
                manager_dict[i] = j
                
    return manager_dict


def loopem(filepath):
    """ Loops through each text file, processes it through the needed functions,
    and returns the split text files.
    """
    
    # Import transcript
    soup = read_file(filepath)
    # Get the body of the text from the call; produce preliminary DataFrame with call details
    body, data = get_attributes(soup)
    # Group call by participants and their text
    call_dict = assign_text_to_speaker(body) # have to modify this function for current transcript 503124
    # Split the call between prepared remarks and q&a sections
    split = prepared_qa_split(body)
    # Assign participant names if call is not formatted to do so automatically
    test_list = []
    for k,v in split['prepared'].items():
        test_list.append(v['group'])
    if 'Corporate Participants' not in test_list:
        call = group_assignment(call_dict, split)
    else:
        call = call_dict
    # Need to modify the split dictionary particiapnt values as well for the group_assignment function
        # and return the split dictionary
    # Group the text by managers and analysts
    new_split = group_managers_analysts(call, split)
    # Group the Q&A section by managers and analysts
    managers_qa, analysts_qa = qa_group_managers_analysts(new_split)
    # Group all prepared text together
    prepared_remarks = new_split['prepared']

    return prepared_remarks, managers_qa, analysts_qa, data


# Remove stop words
def clean(doc):
    """ Removes stop words and any other specified words. Lemmatize each text
    document. Return a normalized text vector.
    """
    
    stop = set(stopwords.words('english') + ['would','thing', 'question','could'])
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = ' '.join([i for i in doc.upper().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    number_free = ''.join([i for i in punc_free if not i.isdigit()])
    normalized = ' '.join(lemma.lemmatize(word.decode('utf-8')) for word in number_free.split())
    
    return normalized


def lda_analysis_bulk(filepath):
    """ Performs an LDA analysis for any given call. Returns topic probability distributions
    for the full prepared text, managers Q&A text, and analysts Q&A text.
    
    Next step is to write additional algorithm to process each response/comment made
    by each participant during the Q&A and benchmark it to the full prepared remarks
    topic probability distribution.
    """
    
    prepared_remarks, managers_qa, analysts_qa, data = loopem(filepath)
    
    # create list of text for prepared remarks to train lda
    prepared_list = []
    for k,v in prepared_remarks.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            prepared_list.append(norm)
    prepared_flat = [item for sublist in prepared_list for item in sublist]
    #pf = list(set(prepared_flat))
    
    # prepare list of text to feed into model; vectorize each set of text; train model    
    dictionary_pre = corpora.Dictionary(prepared_list)
    corpus_pre = [dictionary_pre.doc2bow(doc) for doc in prepared_list]
    lda = LdaModel(corpus_pre, num_topics=5, id2word=dictionary_pre, minimum_probability=.01, passes=1, iterations = 1)
    
    # Predict aggregate text topics for Q&A section by manager/analyst
        # Need to also do in a way that predicts topics for each time speaking for mapping
    manager_list = []
    for k,v in managers_qa.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            manager_list.append(norm)
    manager_flat = [item for sublist in manager_list for item in sublist]
    #mf = list(set(manager_flat))
    
    analyst_list = []
    for k,v in analysts_qa.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            analyst_list.append(norm)
    analysts_flat = [item for sublist in analyst_list for item in sublist]
    #af = list(set(analysts_flat))
    
    qa_flat = list(manager_flat + analysts_flat)
    
    # turn list of prepared remarks text into one block of text; remove duplicate words; vectorize the prepared remarks
    pre_corp = dictionary_pre.doc2bow(prepared_flat)
    qa_corp = dictionary_pre.doc2bow(qa_flat)
    man_corp = dictionary_pre.doc2bow(manager_flat)
    an_corp = dictionary_pre.doc2bow(analysts_flat)
    
    # predict the prepared remarks most discussed topics, in aggregate
    return lda[pre_corp], lda[qa_corp], lda[man_corp], lda[an_corp], data

#-----------------------------------------------------------------------------------------------------------------------------

def lda_analysis_bulk_qa(filepath):
    """ Performs an LDA analysis for any given call. Returns topic probability distributions
    for the full prepared text, managers Q&A text, and analysts Q&A text.
    
    Next step is to write additional algorithm to process each response/comment made
    by each participant during the Q&A and benchmark it to the full prepared remarks
    topic probability distribution.
    """
    
    prepared_remarks, managers_qa, analysts_qa, data = loopem(filepath)
    
    # create list of text for analysts remarks to train lda
    analyst_list = []
    for k,v in analysts_qa.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            analyst_list.append(norm)
    analysts_flat = [item for sublist in analyst_list for item in sublist]
    #analysts_count = len(analysts_flat)
    
    # prepare list of text to feed into model; vectorize each set of text; train model    
    dictionary_pre = corpora.Dictionary(analyst_list)
    corpus_pre = [dictionary_pre.doc2bow(doc) for doc in analyst_list]
    lda = LdaModel(corpus_pre, num_topics=5, id2word=dictionary_pre, minimum_probability=.01, passes=100, iterations = 100)
    
    # Predict aggregate text topics for Q&A section by manager/analyst
        # Need to also do in a way that predicts topics for each time speaking for mapping
    manager_list = []
    for k,v in managers_qa.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            manager_list.append(norm)
    manager_flat = [item for sublist in manager_list for item in sublist]
    #managers_count = len(managers_count)

    prepared_list = []
    for k,v in prepared_remarks.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            prepared_list.append(norm)
    prepared_flat = [item for sublist in prepared_list for item in sublist]    
    
    
    qa_flat = list(manager_flat + analysts_flat)

    # Word Counts for Document
    prepared_count = Counter(prepared_flat)
    qa_count = Counter(qa_flat)
    mqa_count = Counter(manager_flat)
    aqa_count = Counter(analysts_flat)
    # Count prepared remarks
    pcount_df = pd.DataFrame.from_dict(prepared_count, orient='index')
    pcount_df.reset_index(inplace=True)
    pcount_df.columns = ['word','count']
    psum = sum(pcount_df['count'])
    # Count qa 
    qcount_df = pd.DataFrame.from_dict(qa_count, orient='index')
    qcount_df.reset_index(inplace=True)
    qcount_df.columns = ['word','count']
    qsum = sum(qcount_df['count'])
    # Count managers responses
    mqacount_df = pd.DataFrame.from_dict(mqa_count, orient='index')
    mqacount_df.reset_index(inplace=True)
    mqacount_df.columns = ['word','count']
    mqasum = sum(mqacount_df['count'])    
    # Count analysts questions
    aqacount_df = pd.DataFrame.from_dict(aqa_count, orient='index')    
    aqacount_df.reset_index(inplace=True)
    aqacount_df.columns = ['word','count']
    aqasum = sum(aqacount_df['count'])
    
    # Get specific word type counts: total, positive, negative, uncertain, litigious, constraining
    total_count = psum + qsum
    # Managers Responses
    mpos = sum(mqacount_df[mqacount_df['word'].isin(pos_list)]['count'])
    mneg = sum(mqacount_df[mqacount_df['word'].isin(neg_list)]['count'])
    munc = sum(mqacount_df[mqacount_df['word'].isin(uncert_list)]['count'])
    mlit = sum(mqacount_df[mqacount_df['word'].isin(litig_list)]['count'])
    mcon = sum(mqacount_df[mqacount_df['word'].isin(constr_list)]['count'])

    # Analysts Questions
    apos = sum(aqacount_df[aqacount_df['word'].isin(pos_list)]['count'])
    aneg = sum(aqacount_df[aqacount_df['word'].isin(neg_list)]['count'])
    aunc = sum(aqacount_df[aqacount_df['word'].isin(uncert_list)]['count'])
    alit = sum(aqacount_df[aqacount_df['word'].isin(litig_list)]['count'])
    acon = sum(aqacount_df[aqacount_df['word'].isin(constr_list)]['count'])

    # Prepared Remarks
    ppos = sum(pcount_df[pcount_df['word'].isin(pos_list)]['count'])
    pneg = sum(pcount_df[pcount_df['word'].isin(neg_list)]['count'])
    punc = sum(pcount_df[pcount_df['word'].isin(uncert_list)]['count'])
    plit = sum(pcount_df[pcount_df['word'].isin(litig_list)]['count'])
    pcon = sum(pcount_df[pcount_df['word'].isin(constr_list)]['count'])

    # turn list of prepared remarks text into one block of text; remove duplicate words; vectorize the prepared remarks
    pre_corp = dictionary_pre.doc2bow(prepared_flat)
    qa_corp = dictionary_pre.doc2bow(qa_flat)
    man_corp = dictionary_pre.doc2bow(manager_flat)
    an_corp = dictionary_pre.doc2bow(analysts_flat)
    
    # predict the prepared remarks most discussed topics, in aggregate
    return lda[pre_corp], lda[qa_corp], lda[man_corp], lda[an_corp], data, total_count, psum, aqasum, mqasum, mpos, mneg, munc, mlit, mcon, apos, aneg, aunc, alit, acon, ppos, pneg, punc, plit, pcon

#-----------------------------------------------------------------------------------------------------------------------------


def lda_analysis_granular(filepath):
    """ Performs an LDA analysis for any given call. Returns topic probability distributions
    for each response/comment made by each participant during the Q&A.
    
    Next step is to benchmark it to the full prepared remarks topic probability 
    distribution and measure the distance between the vectors.
    """
    
    prepared_remarks, managers_qa, analysts_qa, data = loopem(filepath)
    
    # create list of text for prepared remarks to train lda
    prepared_list = []
    for k,v in prepared_remarks.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            prepared_list.append(norm)
    prepared_flat = [item for sublist in prepared_list for item in sublist]
    pf = list(set(prepared_flat))
    
    # prepare list of text to feed into model; vectorize each set of text; train model    
    dictionary_pre = corpora.Dictionary(prepared_list)
    corpus_pre = [dictionary_pre.doc2bow(doc) for doc in prepared_list]
    lda = LdaModel(corpus_pre, num_topics=5, id2word=dictionary_pre, minimum_probability=.01, passes=100, iterations = 10)
    
    # turn list of prepared remarks text into one block of text; remove duplicate words; vectorize the prepared remarks
    pre_corp = dictionary_pre.doc2bow(pf)
    pre_topics = lda[pre_corp]
    
    # Prepare managers Q&A text for topic prediction analysis
    managers_list = []
    for k,v in managers_qa.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            if len(norm) > 9:
                managers_list.append(norm)
    corpus_managers = [dictionary_pre.doc2bow(doc) for doc in managers_list]
    # predict topics for each managers separate parts of Q&A
    managers_topics = []
    for text in corpus_managers:
        predicted_topics = lda[text]
        managers_topics.append(predicted_topics)
        
    # Prepare analysts Q&A text for topic prediction analysis
    analysts_list = []
    for k,v in analysts_qa.items():
        for i in v['text']:
            norm = clean(i.encode('utf-8')).split()
            if len(norm) > 9:
                analysts_list.append(norm)
    corpus_analysts = [dictionary_pre.doc2bow(doc) for doc in analysts_list]
    # predict topics for each analysts separate parts of Q&A
    analysts_topics = []
    for text in corpus_analysts:
        predicted_topics = lda[text]
        analysts_topics.append(predicted_topics)
    
    return pre_topics, managers_topics, analysts_topics, data


def cosine_distance(filepath):
    """ Calculate the cosine distance between two vectors of topic probability 
    distributions. Creates a DataFrame with three columns: managers distance from the 
    prepared remarks, analysts distance from the prepared remarks, and the
    difference between the managers and analysts distance results.
    
    Inputs must be lists of topic probability distributions returned from the 
    LDA function, i.e. the filepath must contain at least 2 conference calls in
    the folder.
lda[pre_corp], lda[qa_corp], lda[man_corp], lda[an_corp], data, total_count, psum, aqasum, mqasum, mpos, mneg, munc, mlit, mcon, apos, aneg, aunc, alit, acon, ppos, pneg, punc, plit, pcon
    """
    
    prepared = []
    managers = []
    analysts = []
    qa = []
    data = []
    errors = []
    total = []
    pc = []
    aqa = []
    mqa = []
    mpos = []
    mneg = []
    munc = []
    mlit = []
    mcon = []
    apos = []
    aneg = []
    aunc = []
    alit = []
    acon = []
    ppos = []
    pneg = []
    punc = []
    plit = []
    pcon = []
    #for i in filepath:
    try:
	    prepared_i, qa_i, managers_i, analysts_i, data_i, total_i, pc_i, aqa_i, mqa_i, mpos_i, mneg_i, munc_i, mlit_i, mcon_i, apos_i, aneg_i, aunc_i, alit_i, acon_i, ppos_i, pneg_i, punc_i, plit_i, pcon_i = lda_analysis_bulk_qa(filepath)
	    prepared.append(prepared_i)
	    qa.append(qa_i)
	    managers.append(managers_i)
	    analysts.append(analysts_i)
	    data.append(data_i)
	    total.append(total_i)
	    pc.append(pc_i)
	    aqa.append(aqa_i)
	    mqa.append(mqa_i)
	    mpos.append(mpos_i)
	    mneg.append(mneg_i)
	    munc.append(munc_i)
            mlit.append(mlit_i)
            mcon.append(mcon_i)
	    apos.append(apos_i)
	    aneg.append(aneg_i)
	    aunc.append(aunc_i)
	    alit.append(alit_i)
	    acon.append(acon_i)
	    ppos.append(ppos_i)
	    pneg.append(pneg_i)
	    punc.append(punc_i)
	    plit.append(plit_i)
	    pcon.append(pcon_i)
        
	    output_list = []
	    for i in range(len(prepared)):
		topic_df = pd.DataFrame(columns = ['topic'])
		# Specifiy data for the conference call for output DataFrame
		data_i = data[i]
		
		# Prepared vectorization of probabilities in prep for cosine distance calculation
		pi = np.array(prepared[i])
		prepared_df = pd.DataFrame(pi, columns=['topic','probability_prepared'])
		
		# Q&A vectorization of probabilities in prep for cosine distance calculation
		qi = np.array(qa[i])
		qa_df = pd.DataFrame(qi, columns=['topic','probability_qa'])
		
		# Managers vectorization of probabilites in prep for cosine distance calculation
		mi = np.array(managers[i])
		managers_df = pd.DataFrame(mi, columns=['topic','probability_managers'])
		
		# Analysts vectorization of probabilities in prep for cosine distance calculation
		ai = np.array(analysts[i])
		analysts_df = pd.DataFrame(ai, columns=['topic', 'probability_analysts'])
		
		# Merge the three vectors of probabilites together into one DataFrame
		topic_df = topic_df.merge(prepared_df, on='topic', how='outer').fillna(0)
		topic_df = topic_df.merge(qa_df, on='topic', how='outer').fillna(0)
		topic_df = topic_df.merge(managers_df, on='topic', how='outer').fillna(0)
		topic_df = topic_df.merge(analysts_df, on='topic', how='outer').fillna(0)
		topic_df = topic_df.set_index('topic')
		
		# Calculate the cosine distance
		    # smaller number indicates higher similarity in topic discussion relative to prepared remarks
		        # e.g. analysts have small number --> asked more about topics covered in prepared remarks
		        # e.g. managers have small number --> responded with information most similarly discussed in prepared remarks
		qa_cosine_distance = sd.cosine(topic_df['probability_prepared'], topic_df['probability_qa'])
		managers_cosine_distance = sd.cosine(topic_df['probability_prepared'], topic_df['probability_managers'])
		analysts_cosine_distance = sd.cosine(topic_df['probability_prepared'], topic_df['probability_analysts'])
		managers_to_analysts_distance = sd.cosine(topic_df['probability_managers'], topic_df['probability_analysts'])
		# Take the difference between the two distance measures
		    # positive cosine_difference implies analysts discussed more similarly to prepared remarks topics
		        # i.e. managers talked about a broader amount of topics in Q&A than analysts
		    # negative cosine_difference implies managers discussed more similarly to prepared remarks topics
		        # i.e. analysts talked about a broader amount of topics in Q&A than managers
		cosine_difference = managers_cosine_distance - analysts_cosine_distance
		
		# Structure output DataFrame (row vector)
		data_i['qa_distance'] = qa_cosine_distance
		data_i['managers_distance'] = managers_cosine_distance
		data_i['analysts_distance'] = analysts_cosine_distance
		data_i['managers_to_analysts'] = managers_to_analysts_distance
		data_i['distance_difference'] = cosine_difference
		
                data_i['total_count'] = total[i]
		data_i['prepared_count'] = pc[i]
		data_i['questions_count'] = aqa[i]
		data_i['responses_count'] = mqa[i]
		
                data_i['pos_response_count'] = mpos[i]
		data_i['neg_response_count'] = mneg[i]
		data_i['uncertain_response_count'] = munc[i]
		data_i['litigious_response_count'] = mlit[i]
		data_i['constrain_response_count'] = mcon[i]

		data_i['posq_count'] = apos[i]
		data_i['negq_count'] = aneg[i]
		data_i['uncertainq_count'] = aunc[i]
		data_i['litigiousq_count'] = alit[i]
		data_i['constrainq_count'] = acon[i]

		data_i['posp_count'] = ppos[i]
		data_i['negp_count'] = pneg[i]
		data_i['uncertainp_count'] = punc[i]
		data_i['litigiousp_count'] = plit[i]
		data_i['constrainp_count'] = pcon[i]

		output_list.append(data_i)
	    
	    # Build complete DataFrame where each row represents a call's information; return that DataFrame
	    output_df = pd.concat(output_list)
	    
	    return output_df
    except:
	#errors.append(filepath)
	output_df = pd.DataFrame()
	#return errors

""" Additional Intuition on measures (Jason):
I guess I would also interpret the difference measure as the more negative it is 
(conditional on < 0), the more that managers are avoiding the question (i.e.,  
you ask a new question but I dodge it by returning to my script).  If the 
measure > 0, that could either indicate a different approach to dodginess 
(you ask for clarification and instead I start talking about something completely 
new) or it might just indicate that the CEO is relatively loose-lipped and can’t 
stay on script (either his own script or the topic of the analyst question).

Also (maybe, requires more thought), the cosine distance is a better measure of 
the actual level of similarity. We can use the 'distance_difference' as a way 
to sign the 'managers_to_analysts' variable to know who varied in topics more.
"""




path = "/storage/home/sro5/scratch/cc02_14/"
fnames = os.listdir(path)
file_names = [x for x in fnames if 'B' not in x]

file_path = []
for f in file_names:
    file_path.append(path + f)
#del file_path[-2:]

#test = file_path[:500]
#test.append('error_test')
#sample1 = file_path[10000:50000]
#sample2 = file_path[50000:100000]
sample3 = file_path[100000:150000]
#sample4 = file_path[150000:200000]
#sample5 = file_path[200000:250000]
#sample6 = file_path[250000:300000]
#sample7 = file_path[300000:]

start = time.time()
df = Parallel(n_jobs=-1, backend='multiprocessing', max_nbytes=None)(delayed(cosine_distance)(filepath=i) for i in sample3)


out_df = pd.concat(df)
out_df.reset_index(inplace=True, drop=True)
out_df.to_csv('/storage/home/sro5/scratch/cc_data_100-150.csv')




'''
f = open('/storage/home/sro5/scratch/data/conference_call_output/errors_list.txt', 'w')
simplejson.dump(df, f)
f.close()
'''
end = time.time()
elapsed = (end - start)/60
print("Time to completion: " + str(elapsed) + " minutes.")
print("Number of files: " + str(len(file_path)))
print("Number of rows: " + str(len(out_df)))
print("Numer of errors: " + str(len(file_path) - len(out_df)))
'''
data = df[0]

data.reset_index(inplace=True, drop=True)

ssh aci-b.aci.ics.psu.edu
qsub .pbs
qstat -u sro5
'''









