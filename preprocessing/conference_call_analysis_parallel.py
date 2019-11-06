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
#import simplejson
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
#import gensim
#from gensim import corpora
#LdaModel = gensim.models.ldamodel.LdaModel
import scipy.spatial.distance as sd
import os
import time
from joblib import Parallel, delayed
import multiprocessing
from collections import Counter
from preprocessing.graph import DocGraph
from documents.document import Document

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


def assign_attendee(d, attendees, doc_graph=None):
    """ Fills dictionary with speakers role, job, and respective text (empty). """
    
    new_d = defaultdict(dict)
    prev_a = ''
    prev_doc = ''

    for key, doc in d.items():
        name_split = key.split("[")
        a = name_split[0].strip().split(",")[0].strip()
        #doc_idx = int(name_split[-1].split("]")[0].strip())
        if a not in attendees:
            continue

    # Add in job and participant type info
        if a not in new_d:
            new_d[a]['doc_idx'] = []
            new_d[a]['text'] = []
        if a in attendees:
            new_d[a]['job'] = attendees[a][0]
            new_d[a]['group'] = attendees[a][1]
        # If person doesn't appear in list of participants, we don't have info on job/group
        else:
            new_d[a]['job'] = None
            new_d[a]['group'] = None

        new_d[a]['text'].append(doc)
        #new_d[a]['doc_idx'].append(doc_idx)


        if doc_graph is not None:
            # build edge list
            if prev_a in new_d and new_d[prev_a]['group'] == 'Conference Call Participants' and new_d[a]['group'] == 'Corporate Participants':
                # add edge
                doc_graph.add_edge(prev_doc, doc)

            document = doc_graph.get_doc(doc)
            if document is not None:
                document.set_speaker_type(new_d[a]['group'])
                document.set_speaker(a)


        prev_a = a
        prev_doc = doc


    return new_d


def assign_text_to_speaker(body, doc_graph):
    """ Fills values of dictionary with speakers role, job, and respective text. 
    The full call transcript is populated throughout the values of the dictionary.
    """
    
    sections = subdivide(body, "^=+")
    # regex pattern for matching headers of each section
    header_pattern = re.compile("^.*[^\n]", re.MULTILINE)

    # regex pattern for matching the sections that contains
    # the list of attendee's (those that start with asterisks )
    #if unicode("Corporate Participants", "utf-8") in sections:
    ppl_pattern = re.compile("^(\s+\*)(.+)(\s.*)", re.MULTILINE)
    #else:
    #    ppl_pattern = re.compile("^(\s+\*)(\s.*)", re.MULTILINE)
        
    # regex pattern for matching sections with subsections in them.
    dash_pattern = re.compile("^-+", re.MULTILINE)

    ppl_d = dict()
    #ppl_d['Operator'] = ['Operator', 'Operator']
    talks_d = dict()

    header = []
    # Step2. Handle each section like a switch case
    for section in sections:
        # Handle headers
        if len(section.split('\n')) == 1:  # likely to match only a header (assuming )
            header = header_pattern.match(section).string
    
        # Handle attendees/presenters
        elif ppl_pattern.match(section):
            #if unicode("Corporate Participants", "utf-8") in sections:
            ppls = ppl_pattern.findall(section)
            d = {key.strip(): value.strip() for (_, key, value) in ppls}
            #else:
            #    ppls = ppl_pattern.findall(section)
            #    ppls_list = []
            #    for i in ppls:
            #        val = unicode('particiapnt', 'utf-8')
            #        ppls_new = i + (val,)
            #        ppls_list.append(ppls_new)
            #    d = {key.strip(): value.strip() for (_, key, value) in ppls_list}
            
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
            for speaker, text in d.items():
                if 'operator' in speaker.lower():
                    continue
                else:
                    doc = Document(text=text)
                    doc_graph.add_node(doc)
            
        # Else its just some random text.
        else:
    
            # assuming that if the previous section was detected as a header, then this section will relate
            # to that header
            if header:
                talks_d.update({header: section})            

    # To assign the talks material (as a list) to the appropriate attendee/presenter. Still works if no match found.
    for key, value in talks_d.items():
        talks_d[key] = assign_attendee(value, ppl_d, doc_graph)
    
    return talks_d, doc_graph


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
    #if unicode("Corporate Participants", "utf-8") in sections:
    ppl_pattern = re.compile("^(\s+\*)(.+)(\s.*)", re.MULTILINE)
    #else:
    #    ppl_pattern = re.compile("^(\s+\*)(\s.*)", re.MULTILINE)

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
            #if unicode("Corporate Participants", "utf-8") in sections:
            ppls = ppl_pattern.findall(section)
            d = {key.strip(): value.strip() for (_, key, value) in ppls}
            #else:
                #ppls = ppl_pattern.findall(section)
                #ppls_list = []
                #for i in ppls:
                #    val = unicode('particiapnt', 'utf-8')
                #    ppls_new = i + (val,)
                #    ppls_list.append(ppls_new)
                #d = {key.strip(): value.strip() for (_, key, value) in ppls_list}
            
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
            else:
                continue
            #doc_graph.add_node(j['text'])
            
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
        else:
            continue



        
    return manager_dict, analyst_dict


def get_manager_text(new_split_dictionary):
    """ Function to collect all manager text into one dictionary. """
    
    manager_dict = {}
    for k,v in new_split_dictionary.items():
        for i,j in v.items():
            if j['group'] == 'Corporate Participants':
                manager_dict[i] = j
                
    return manager_dict


def loopem(filepath, doc_graph):
    """ Loops through each text file, processes it through the needed functions,
    and returns the split text files.
    """
    
    # Import transcript
    soup = read_file(filepath)
    # Get the body of the text from the call; produce preliminary DataFrame with call details
    body, data = get_attributes(soup)
    # Group call by participants and their text
    call_dict, doc_graph = assign_text_to_speaker(body, doc_graph) # have to modify this function for current transcript 503124
    # Split the call between prepared remarks and q&a sections
    """split = prepared_qa_split(body)
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

    # add remaining edges
    try:
        DocGraph.build_graph(doc_graph, call_dict['Presentation'], call_dict['Questions and Answers'])
    except KeyError:
        pres_dict = list(call_dict.keys())[0]
        qq_dict = list(call_dict.keys())[1]
        DocGraph.build_graph(doc_graph, call_dict[pres_dict], call_dict[qq_dict])"""

    return doc_graph


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
    normalized = ' '.join(lemma.lemmatize(word) for word in number_free.split())
    
    return normalized






