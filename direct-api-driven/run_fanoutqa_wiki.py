import os
import json
import re
from langchain_community.utilities import SerpAPIWrapper, GoogleSerperAPIWrapper
from autogpt import AutoGPT
from langchain.agents import Tool
from langchain_openai import ChatOpenAI
from langchain_community.docstore import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks import get_openai_callback
from tqdm import tqdm
import faiss
from urllib.parse import unquote
from bs4 import BeautifulSoup
from markdownify import markdownify as md
import requests
from typing import List, Dict
from cachetools import TTLCache, cached
import tiktoken
import functools
import argparse
from copy import deepcopy
import time
import datetime
import httpx
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import Tool, tool


DATASET_EPOCH = datetime.datetime(year=2023, month=11, day=20, tzinfo=datetime.timezone.utc)
"""The day before which to get revisions from Wikipedia, to ensure that the contents of pages don't change over time."""

USER_AGENT = "fanoutqa/1.0.0 (andrz@seas.upenn.edu)"

# Initialize sentence transformer once at module level
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

class BFSAgent:
    def __init__(self, args) -> None:
        print(f"\nInitializing BFSAgent with deployments:")
        print(f"Navigator Model: {args.navigator_model}")
        print(f"Chat Deployment: {args.chat_deployment}")
        print(f"Embedding Deployment: {args.embedding_deployment}")
        
        self.navigator_model = args.navigator_model
        self.aggregator_model = args.aggregator_model
        self.extractor_model = args.extractor_model
        self.chat_deployment = args.chat_deployment
        self.embedding_deployment = args.embedding_deployment
        
        self.enc = tiktoken.get_encoding("cl100k_base")

        self.aggregator_messages = list()

        self.num_iterations = args.num_iterations
        self.num_to_aggregate = args.num_to_aggregate
        self.timeout = 10

        self.wikipedia = httpx.Client(
            base_url="https://en.wikipedia.org/w/api.php",
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=30,
        )
        self.wiki_errors = list()

        self.aggregated_data = list()
        self.counter = 1
        self.extractor_time = 0.0
        self.aggregator_time = 0.0
        self.search_time = 0.0
        self.parse_time = 0.0

        self.current_query = None
        self.last_search_thought = "Not Available"
        self.last_selection_thought = "Not Available"
        self.previous_iterations = list()

        self.init_nav_agent()

    def setup(self, user_task):
        self.user_task = user_task

    def init_nav_agent(self):
        """Initialize the navigation agent with AutoGPT"""
        tools = [
            Tool(
                name="search",
                func=self.search,
                description="Useful for searching Wikipedia articles. Returns relevant articles with their titles and snippets."
            ),
            Tool(
                name = "extract",
                func=self.extract,
                description="Useful to extract relevant information from provided URL and get feedback from Aggregator Module on how to proceed next"
            )
        ]

        # Initialize vector store using the global embeddings model
        embedding_size = 768
        index = faiss.IndexFlatL2(embedding_size)
        vectorstore = FAISS(EMBEDDINGS_MODEL, index, InMemoryDocstore({}), {})

        # Initialize the agent LLM
        agent_llm = ChatOpenAI(
            temperature=0.7,
            model_name="meta-llama/Llama-3.1-70B-Instruct",
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            max_retries=3
        )

        # Create the AutoGPT agent with updated initialization
        self.agent = AutoGPT.from_llm_and_tools(
            ai_name="BFSAgent",
            ai_role="Assistant",
            tools=tools,
            llm=agent_llm,
            memory=vectorstore.as_retriever(),
            human_in_the_loop=False
        )
        
        # Set verbose to be false
        self.agent.chain.verbose = False

    def extract_info_prompt(self, task, data):
        print(f"\nAttempting Extractor LLM call with deployment: {self.chat_deployment}")
        extract_start = time.time()
        encoded = self.enc.encode(data)
        print("Length of content: ", len(encoded))
        print(self.last_search_thought)
        print(self.last_selection_thought)

        # Need to truncate input to OpenAI call 
        if len(encoded) > 100000:
            data = self.enc.decode(encoded[:100000])

        prompt = """WEBSITE CONTENT: {data}\n\n\n
        The above website was chosen with the following motivation by the web navigator when searching the web: {search_motivation}
        
        You must follow the navigator's current motivation to identify the relevant information from the provided content. Note that the relevant information can directly or even partially address what the navigator is looking or even provide more context to later navigator searches. You must return the relevant information in the form of a list of paragraphs. Each paragraph should NOT be longer than 8 sentences.\n You can return upto two paragraphs ONLY. 

        Your goal is to be an EXTRACTOR, i.e. you must only identify and extract relevant facts from the website content. However, you MUST NOT draw any inferences on your own based on the website content or directly try to answer the navigator's motivation. You can choose to merge or combine these facts as needed since you can return only two paragraphs. 
        
        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads.
        Response Format: 
        {{ 
            "thoughts": "Concise justification for why the information was deemed relevant",
            "paragraphs": [list of relevant paragraphs according to the navigator's motivation]
        }}"""

        inp_prompt = prompt.format(data=data, search_motivation=self.last_search_thought)
        messages = [
            {"role": "system", "content": """You are an assistant helping an information aggregation process. You are working with a web navigator that iteratively searches for information. Your goal is identify and extract any relevant information from the website content that the web navigator has provided to you in the current iteration. 
            
            You must respond in valid JSON format with the following structure:
            {
                "thoughts": "your reasoning",
                "paragraphs": ["paragraph1", "paragraph2"]
            }
            
            DO NOT RESPOND with your own knowledge, only respond BASED on information provided in the text. Current date: 11-20-2023."""},
            {"role": "user", "content": inp_prompt}
        ]
        
        llm = ChatOpenAI(
            model=self.chat_deployment,
            max_tokens=2000, 
            temperature=0
        )
        
        response = llm.invoke(messages)
        print(response)      
        self.extractor_time += time.time() - extract_start

        try:
            data = json.loads(response.content)
            if type(data).__name__ == "dict":
                if len(data) == 2:
                    value = data["paragraphs"]
                    if type(value).__name__ == "list":
                        return value, data, False
                    else:
                        return [value], data, False
                else:
                    return [data]
            elif type(data).__name__ == "list":
                return data, data, False
            else:
                return [data], data, False
        except:
            return [response.content], response.content, True

    def wikipedia_to_markdown(self, title: str) -> str:
        """Convert Wikipedia page content to markdown format using markdownify, removing CSS styles."""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "parse",
                "page": title,
                "format": "json",
                "prop": "text",
                "origin": "*"  # Avoids potential CORS issues
            }
            response = requests.get(url, params=params)
            response.raise_for_status()

            # Retrieve HTML content from the JSON response
            html_content = response.json().get("parse", {}).get("text", {}).get("*", "")
            if not html_content:
                return ""

            # Remove <style>...</style> blocks from the HTML to eliminate unwanted CSS
            cleaned_html = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)

            # Convert the cleaned HTML to markdown
            markdown_text = md(cleaned_html)
            return markdown_text

        except Exception as e:
            print(f"Error converting Wikipedia page to markdown: {e}")
            return ""

    def search(self, query: str) -> List[Dict]:
        """Search Wikipedia for information using the MediaWiki API."""
        print("\n=====================================")
        print("SEARCH METHOD CALLED DIRECTLY")
        print(f"Query received: {query}")
        print("=====================================\n")
                   
        self.last_search_thought = f"Searching for: {query}"
        print(f"\n=== Wikipedia Search Request ===")
        print(f"Query: {query}")
        
        search_start = time.time()
        try:
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "origin": "*",
            }
            print(f"Making API request to: {search_url}")
            print(f"With params: {params}")
            
            response = requests.get(search_url, params=params)
            print(f"Response status code: {response.status_code}")
            if response.status_code != 200:
                print(f"Search failed with status code: {response.status_code}")
                print("Response content:", response.text)
            
            snippets = []
            if response.status_code == 200:
                search_results = response.json().get("query", {}).get("search", [])
                print(f"Total results found: {len(search_results)}")
                
                for i, result in enumerate(search_results[:3]):
                    title = result["title"]
                    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
                    snippet = {
                        "title": title,
                        "url": url,
                        "snippet": " ".join(self.wikipedia_to_markdown(title).split()[:250])
                    }
                    snippets.append(snippet)
                    # print(f"\nResult {i+1}:")
                    # print(f"Title: {snippet['title']}")
                    print(f"URL: {snippet['url']}")
                    # print(f"Snippet: {snippet['snippet'][:100]} ... {snippet['snippet'][-100:]}")
            
            self.search_time += time.time() - search_start
            # print(f"\nSearch completed in {time.time() - search_start:.2f} seconds")
            self.current_query = query
            
            return snippets
            
        except Exception as e:
            print(f"\n=== Search Error ===")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            print("==================\n")
            return []

    def aggregator(self, data):
        print(f"\nAttempting Aggregator LLM call with deployment: {self.chat_deployment}")
        aggregator_start = time.time()
        if len(self.aggregated_data) > 0:
            aggregated_list = ""
            for pid, text in enumerate(self.aggregated_data):
                aggregated_list += "ID: E" + str(pid+1) + " Text: " + text + "\n\n"
        else:
            aggregated_list = "None"

        provided_list = ""
        for pid, text in enumerate(data):
            provided_list += "ID: P" + str(pid+1) + " Text: " + text + "\n\n"

        previous_iterations = ""
        for c, iter in enumerate(self.previous_iterations):
            previous_iterations += """Iteration {i}: Thoughts - {t}\nFeedback - {f}\n\n""".format(i=str(c+1), t=iter["thoughts"], f=iter["feedback"])

        system_prompt = """You are an information aggregation assistant designed to aggregate relevant information across multiple iterations for a given user query. Your goal at every iteration is to identify any relevant information that can help answer user query. If the information does not DIRECTLY answer the user query, you can still consider the information if it provides context that is useful for later iterations in order to answer the user query. Make sure to not gather information that is duplicate, i.e. do not add redudant information into what you have already aggregated. You should stop aggregating when you have enough information to answer the user query. REMEMBER that your goal is ONLY to aggregate relevant information and NOT to generate the FINAL answer to the user query. Current date: 11-20-2023."""

        system_input = system_prompt

        prompt = """You will be provided with a set of passages collected from a website by a navigator assistant. You need to decide whether any of the provided information should be added into the aggregated information list. You can choose to add the information if it can address the user query. If the information does not DIRECTLY address the user query, you can still aggregate the information if it provides context that is useful for later iterations in order to answer the user query. You also have the option to ignore and not aggregate any of the provided passages into the aggregated information list. 
        
        Further, you should provide feedback to the navigator assistant on what specific information to look for next. If the information needs to be gathered in multiple steps, you can break it down to multiple steps and sequentially instruct the navigator. The navigator assistant cannot see the information aggregated so far, so you make sure the feedback is clear and does not have any ambiguity. Make sure to refer to any entities by their full names when providing the feedback so that the navigator knows what you are referring to. You should instruct the navigator to terminate if enough information to answer the user query has been aggregated.

        When you have enough information to answer the user query, you must return the aggregated information. Make sure to return the aggregated information not to say you have enough information.

        You have a maximum of {num_iterations} iterations overall, after which the information aggregated will be automatically returned. You are also provided access to your thoughts and navigator feedback from previous iterations, so that you don't repeat yourself. In addition, the navigator's thoughts are also shown to give additional context for why the provided information was extracted.
        
        Current Iteration Counter: {counter}

        User Task: {user_task}

        Previous Iterations: {previous_iterations}

        Navigator Thoughts: {navigator_motivation}

        Information Aggregated so far: 
        {aggregated_list}

        Provided information: 
        {provided_list}  

        You should only respond in JSON format as described below and ensure the response can be parsed by Python json.loads
        Response Format: 
        {{
            "thoughts": "Your step-by-step reasoning for what actions to perform based on the provided information",
            "actions": [list of actions (generated as a string) to perform. Allowed actions are: REPLACE(existing_id, provided_id) if passage existing_id in aggregated information should be replaced by passage provided_id from provided information and ADD(provided_id) if passage provided_id should be added to aggregated information],
            "feedback": "Feedback to return to the navigator assistant n what specific information to look for next. The navigator assistant does not have access to the information aggregated, so be clear in your feedback. Also let the navigator assist know how many more iterations are left."
        }}"""
        inp_prompt = prompt.format(num_iterations=str(self.num_iterations), counter=str(self.counter), user_task=self.user_task, previous_iterations=previous_iterations, navigator_motivation=self.last_search_thought, aggregated_list=aggregated_list, provided_list=provided_list)

        messages = [
            {"role": "system", "content": system_input + """
            You must respond in valid JSON format with the following structure:
            {
                "thoughts": "your reasoning",
                "actions": ["action1", "action2"],
                "feedback": "your feedback"
            }"""},
            {"role": "user", "content": inp_prompt}
        ]
        
        llm = ChatOpenAI(
            model=self.chat_deployment,
            max_tokens=1000, 
            temperature=0.7
        )
        
        response = llm.invoke(messages)
        self.aggregator_time += time.time() - aggregator_start

        try:
            output = json.loads(response.content)
            self.previous_iterations.append({
                "thoughts": output["thoughts"],
                "feedback": output["feedback"]
            })
            print(output)
            for action in output["actions"]:
                try:
                    if "REPLACE" in action:
                        pattern = r'^REPLACE\(E(\d+),\s*P(\d+)\)$'
                        match = re.match(pattern, action)
                        if match:
                            integer1, integer2 = match.groups()
                            self.aggregated_data[int(integer1)-1] = data[int(integer2) - 1]
                        else:
                            print("Invalid Action: ", action)

                    if "ADD" in action:
                        pattern = r"^ADD\(P(-?\d+)\)$"
                        match = re.match(pattern, action)
                        if match:
                            pindex = int(match.group(1)) - 1
                            self.aggregated_data.append(data[pindex])
                        else:
                            print("Invalid Action: ", action)
                except Exception as e:
                    print(e)
            
            return output["feedback"] + " Passages aggregated so far: " + str(len(self.aggregated_data)), output, False
        except:
            return "Aggregator failed to parse the website content. Continue with a different website.", response.content, True
        
    @cached(cache=TTLCache(maxsize=10000, ttl=6000)) 
    def get_revid(self, pageid):
        resp = self.wikipedia.get(
            "",
            params={
                "format": "json",
                "action": "query",
                "prop": "revisions",
                "rvprop": "ids|timestamp",
                "rvlimit": 1,
                "pageids": pageid,
                "rvstart": DATASET_EPOCH.isoformat(),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        page = data["query"]["pages"][str(pageid)]
        return page["revisions"][0]["revid"]
    
    @cached(cache=TTLCache(maxsize=10000, ttl=6000)) 
    def reset_to_old_version(self, url):       
        try:
            title = url.split("/wiki/")[-1]
            parsed_title = unquote(title)
            resp = self.wikipedia.get("", params={"format": "json", "action": "query", "titles": parsed_title})
            resp.raise_for_status()
            data = resp.json()
            pages = data['query']['pages']
            if len(pages) > 1:
                return url
            else:
                page_id = next(iter(pages))
                try:
                    revid = self.get_revid(page_id)
                    return f'https://en.wikipedia.org/w/index.php?title={title}&oldid={revid}'
                except Exception as e:
                    self.wiki_errors.append({
                        "type": "revision",
                        "parsed_title": parsed_title,
                        "page_id": page_id,
                        "title": title
                    })
                    print(str(e))
                    return url
        except Exception as e:
            self.wiki_errors.append({
                        "type": "page",
                        "parsed_title": parsed_title,
                        "title": title
                    })
            print(str(e))
            return url
        
    def extract(self, url: str) -> List[str]:
        # Don't try to invoke the agent directly for thoughts
        self.last_selection_thought = f"Extracting from: {url}"

        self.counter += 1
        parse_start = time.time()
        """Useful to extract relevant information from the given url for the current user task"""
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
        url = self.reset_to_old_version(url)
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            html = response.content
            # text = BeautifulSoup(html, 'html.parser').get_text()
            # cleaned_text = re.sub(r'\n+', '\n', text)
            content = md(html, heading_style="ATX")
        except Exception as e:
            print(str(e))
            content = ""

        print("URL parsing done")
        self.parse_time += time.time() - parse_start

        if content.strip():
            extracted_data, extractor_log, extractor_parse_error = self.extract_info_prompt(self.user_task, content)
            print("Extracted Data: ", extracted_data)
            if len(extracted_data) > 0:
                aggregator_feedback, aggregator_log, aggregator_parse_error = self.aggregator(extracted_data)
            else:
                aggregator_feedback = "Did not find any relevant information on the website. Continue collecting information." + " Passages aggregated so far: " + str(len(self.aggregated_data)) 
                aggregator_log = None
                aggregator_parse_error = False
        else:
            aggregator_feedback =  "Did not find any relevant information on the website. Continue collecting information." + " Passages aggregated so far: " + str(len(self.aggregated_data)) 
            aggregator_log = None
            aggregator_parse_error = False
            extractor_parse_error = False
            extractor_log = None
        
        self.aggregator_messages.append({
            "url": url,
            "extractor_log": extractor_log,
            "extractor_parse_error": extractor_parse_error,
            "aggregator_log": aggregator_log,
            "aggregator_parse_error": aggregator_parse_error
        })

        return aggregator_feedback

    def run(self):
        output = dict()

        prompt = """You are an assistant aiding an information aggregation process designed to gather information that can answer a multi-hop user question. 
        
        You are provided access to the web (using the "search" command) which returns websites relevant to your search query. Based on the provided websites, you should then choose to visit (using the "extract" command) website that is most relevant. Along with the website URL, you are also provided with a short snippet from the website that can help to decide whether the website is relevant to visit. 
        
        You should only visit websites that you think will contain information relevant to user query. If the websites do not contain any relevant information, you can choose to perform a different search. DO NOT visit a website that you have already have visited before. Note that information cannot be directly aggregated based on the search command. You MUST ALSO visit the website using the extract command in order to be able to aggregate the relevant information from it. 
        
        You will work in conjunction with an aggregator assistant (which runs as part of the "extract" command) that keeps track of information aggregated and will give feedback to you on what information to look for next. You can decide to stop if aggregator assistant tells you so or if you keep running into a loop. You can simply terminate at the end with a message saying aggregation is done.

        Note that you can only search for one piece of information at a time. For the multi-hop query, it is good to break it down and gather relevant information sequentially over multiple iterations. If the extract command suggests you to search for multiple pieces of information, you should search for each piece sequentially over different iterations.

         Make sure to use "extract" command after searching for the most relevant websites! The queries should be entity based as used in wikipedia.

        Current date: 11-20-2023.
        Query: {task}

        Let's approach this step by step. Start by searching for the most relevant information first."""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Run the agent with the formatted prompt
                result = self.agent.run(goals=[prompt.format(task=self.user_task)])
                output["navigator_output"] = result
                break  # If successful, break out of retry loop
            except Exception as e:
                if "DeploymentNotFound" in str(e):
                    if attempt == max_retries - 1:  # Last attempt
                        print(f"Error: Llama deployment not found after {max_retries} retries.")
                        output["navigator_output"] = "Error: Llama deployment configuration issue"
                    else:
                        print(f"Deployment not found, retrying... (attempt {attempt+1}/{max_retries})")
                        time.sleep(5)  # Wait 5 seconds before retrying
                        continue
                else:
                    print(f"Agent run error: {str(e)}")
                    output["navigator_output"] = str(e)
                    break  # Break on non-deployment errors

        output["aggregated_output"] = self.aggregated_data
        output["langchain_messages"] = []
        if hasattr(self.agent, 'chat_history_memory'):
            for m in self.agent.chat_history_memory.messages:
                output["langchain_messages"].append(m.content)

        output["aggregator_messages"] = self.aggregator_messages

        return output

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--navigator_model",
        help="Model to use for navigation",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--aggregator_model",
        help="Model to use for aggregation",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--extractor_model",
        help="Model to use for extraction",
        default="gpt-4o-mini",
        type=str
    )
    parser.add_argument(
        "--num_iterations",
        help="Number of iterations to run the aggregator for",
        default=10,
        type=int
    )
    parser.add_argument(
        "--num_to_aggregate",
        help="Number of passages to aggregate",
        default=5,
        type=int
    )
    parser.add_argument(
        "--inp_path",
        help="Input file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--out_path",
        help="Output file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--log_path",
        help="Log file path",
        required=True,
        type=str
    )
    parser.add_argument(
        "--chat_deployment",
        help="OpenAI chat deployment name",
        required=True,
        type=str
    )
    parser.add_argument(
        "--embedding_deployment", 
        help="OpenAI embedding deployment name",
        required=True,
        type=str
    )
    args = parser.parse_args()

    starting_index = 58
    data = json.load(open(args.inp_path))
    data = data[:starting_index]
    output = list()
    log = list()
    for item in tqdm(data):
        bfsagent = BFSAgent(args)
        bfsagent.setup(item["question"])
        try:
            time_start = time.time()
            output_item = bfsagent.run()
            total_time = time.time() - time_start
            output_item["time"] = {
                "total_time": total_time,
                "navigator_time": total_time - (bfsagent.search_time + bfsagent.parse_time + bfsagent.extractor_time + bfsagent.aggregator_time),
                "extractor_time": bfsagent.extractor_time,
                "aggregator_time": bfsagent.aggregator_time,
                "search_time": bfsagent.search_time,
                "parse_time": bfsagent.parse_time
            }
            output_item["id"] = item["id"]
            output_item["wiki_errors"] = deepcopy(bfsagent.wiki_errors)
            output.append(deepcopy(output_item))
            
            # Save output after each item
            json.dump(output, open(args.out_path, "w"), indent=4)
            
        except Exception as e:
            error_entry = {
                "id": item["id"],
                "error": str(e)
            }
            log.append(error_entry)
            # Save log after each error
            json.dump(log, open(args.log_path, "w"), indent=4)
            continue
    
    # Final save of both files (in case the last save was skipped)
    json.dump(output, open(args.out_path, "w"), indent=4)
    json.dump(log, open(args.log_path, "w"), indent=4)