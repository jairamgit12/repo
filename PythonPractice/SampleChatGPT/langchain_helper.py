import os
from secret_key import openai_key
os.environ['OPENAI_API_KEY'] = openai_key
 
from langchain_community.llms import OpenAI
# from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

def about_the_topic(topic):
    llm = OpenAI(temperature=0.6)
 
    # topic chain
    prompt_template = PromptTemplate(
    input_variables = ['topic'],
    template = "About {topic} ."
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="Topic")
  
    # Sequential chain
    chain = SequentialChain(
    chains = [name_chain],
    input_variables = ['topic']
    )
 
    response = chain({'topic' : topic})
    return response