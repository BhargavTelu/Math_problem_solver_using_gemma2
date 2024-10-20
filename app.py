import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMMathChain,LLMChain
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Set up the streamlit app
st.set_page_config(page_title="Math problem solver using google Gemma", page_icon="🦜️")
st.title("🦜️ Math problem solver")

#get the groq api key and url
groq_api_key = st.sidebar.text_input("GROQ_API_KEY", type = "password", value="")
if not groq_api_key:
    st.info("please provide Groq API KEY")
    st.stop()
llm = ChatGroq(groq_api_key = groq_api_key , model_name = "Gemma2-9b-It", streaming=True)

#initialize the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="tool for searching the internet"
)

# initialize the math tool
math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(name = "calculator",
                func=math_chain.run,
                description="A tool for answering math related questions")

#defining prompt template
prompt="""You are a agent tasked for solving mathematical question. Logically arrive at the solution and display it point wise.
Question:{question}
Answer:
"""
prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)

#combine all the tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)
reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logical based questions"
)

# initializing the agents
assistant_agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am a math chatbot who can answer the math questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"]) # appending all messages in chat_messages

# function to generate the response
# def generate_response(question):
#     response=assistant_agent.invoke({"input":question})
#     return response

question=st.text_area("Enter your question:")
if st.button("Find the solution"):
    if question:
        with st.spinner("Generate response.."):
            st.session_state.messages.append({"role":"user", "content":question})
            st.chat_message("user").write(question) 
            streamlit_callback=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[streamlit_callback])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("## response")  
            st.success(response)   
    else:
        st.warning("please enter a question")        
