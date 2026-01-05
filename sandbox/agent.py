# --- START OF FILE sandbox/agent.py ---

from langchain_openai import OpenAIEmbeddings
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_community.vectorstores import FAISS
from openai import OpenAI, AsyncOpenAI

from sandbox.log import Log, PlanningLog, ConclusionLog
from sandbox.message import AgentMessage
from sandbox.tool import call_tool, call_hijack_tool
from sandbox.pre_info import AgentInfo
from sandbox.action import Action
from sandbox.prompt import THINK_TEMPLATE
from sandbox.prompt import COT_PROMPT

import random
import yaml
import json
import os
import re
import shutil
from typing import List, Tuple
import asyncio
import aiofiles
import time

from sandbox.utils import (
    get_embedding,
    cosine_similarity,
    clean_json_string
)

PRECOMPUTED_FAISS_INDEX_DIR = './Vector_DB/precomputed_index/'

def _normalize_base_url(u: str) -> str:
    """Normalizes the OpenAI-compatible base URL to ensure proper formatting."""
    u = (u or "").strip()
    if not u:
        return u

    u = u.rstrip("/")

    for suffix in [
        "/chat/completions",
        "/v1/chat/completions",
        "/responses",
        "/v1/responses",
    ]:
        if u.endswith(suffix):
            u = u[: -len(suffix)].rstrip("/")
            break

    if u.endswith("/v1"):
        return u
    return u + "/v1"


class Agent:
    """
    Base class for a generic agent in the simulation.
    Handles memory management, tool usage, RAG interaction, and communication.
    """

    def __init__(
        self,
        id,
        name: int,
        model: dict,
        tools: list[str],
        background: AgentInfo,
        simulator,
        use_rag: bool,
        extra_data,
        poison_rag: bool = False,
        max_memory: int = 5,
        log_path: str = None,
        hijack_tool: bool = False
    ) -> None:
        self.id = id
        self.name = name
        self.model = model
        self.tools = tools
        self.background = background
        self.hijack_tool = hijack_tool
        self.short_term_memory = []
        self.max_memory = max_memory
        self.message_buffer = []
        self.received_messages = ""
        self.conversation_buffer = AgentMessage(-1, -1, "")
        self.simulator = simulator
        self.use_rag = use_rag
        self.extra_data = extra_data
        self.rag_dir = f"./Vector_DB/vectorstore_agent_{self.name}/"
        self.embedding = None
        self.log_path = log_path
        self.long_memory = ""

        try:
            self.model["base_url"] = _normalize_base_url(self.model.get("base_url"))
        except Exception:
            pass

        try:
            self.aclient = AsyncOpenAI(
                api_key=self.model["api_key"],
                base_url=self.model["base_url"],
                timeout=60.0,
                max_retries=3
            )
        except Exception as e:
            print(f"Error: Agent {self.name} failed to initialize AsyncOpenAI: {e}")
            self.aclient = None

        if self.use_rag:
            os.makedirs(self.rag_dir, exist_ok=True)
            try:
                with open("config/api_keys.yaml", "r", encoding="utf-8") as f:
                    config = yaml.load(f, Loader=yaml.FullLoader) or {}

                openai_api_key = config.get("openai_api_key")
                openai_base_url = config.get("openai_base_url")

                if not openai_api_key:
                    self.use_rag = False
                else:
                    embedding_params = {"openai_api_key": openai_api_key}

                    if openai_base_url:
                        openai_base_url = _normalize_base_url(openai_base_url)
                        embedding_params["base_url"] = openai_base_url

                    embed_model = (
                        config.get("embedding_model")
                        or os.environ.get("EMBEDDING_MODEL")
                        or "text-embedding-3-small"
                    )

                    candidates = []
                    for m in [embed_model, "text-embedding-3-small", "text-embedding-ada-002", "text-embedding-3-large"]:
                        if m and m not in candidates:
                            candidates.append(m)

                    self.embedding = None
                    for m in candidates:
                        try:
                            embedding_params["model"] = m
                            self.embedding = OpenAIEmbeddings(**embedding_params)
                            break
                        except Exception:
                            continue

                    if self.embedding is None:
                        self.use_rag = False

                if self.use_rag:
                    precomputed_index_faiss = os.path.join(PRECOMPUTED_FAISS_INDEX_DIR, "index.faiss")
                    precomputed_index_pkl = os.path.join(PRECOMPUTED_FAISS_INDEX_DIR, "index.pkl")

                    if not os.path.exists(precomputed_index_faiss) or not os.path.exists(precomputed_index_pkl):
                        self.use_rag = False
                    else:
                        shutil.copy2(precomputed_index_faiss, self.rag_dir)
                        shutil.copy2(precomputed_index_pkl, self.rag_dir)
                        main_db = FAISS.load_local(
                            self.rag_dir,
                            self.embedding,
                            allow_dangerous_deserialization=True
                        )

                        if poison_rag and isinstance(self.extra_data, dict) and "misinfo_argument" in self.extra_data:
                            poison_docs = [
                                Document(page_content=f"{self.extra_data.get('user_input','')}{doc}")
                                for doc in (self.extra_data.get("misinfo_argument") or [])
                            ]
                            if poison_docs:
                                main_db.add_documents(poison_docs)
                                main_db.save_local(self.rag_dir)

            except Exception as e:
                print(f"Agent {self.name} RAG initialization error: {e}")
                self.use_rag = False

    async def close_client(self):
        if hasattr(self, 'aclient') and self.aclient:
            try:
                await self.aclient.close()
            except: pass
            self.aclient = None

    def add_background(self, agent_message, background, short_term_memory, rag_dir) -> AgentMessage:
        self.long_memory = ""
        if self.use_rag:
            rag_docs = self._search_rag(self.background.data["user_input"] + agent_message.prompt, num=5)
            for index, rag_doc in enumerate(rag_docs, start=1):
                self.long_memory += f"--- Reference {index} ---\n{rag_doc.page_content}\n"

        des_thought = (
            f"**Background:** {background.info}\n"
            f"**Neighbors:** {background.neighbors}\n"
            f"**Tools:** {self.tools}\n"
        )
        if len(short_term_memory):
            for index, short_term in enumerate(short_term_memory):
                des_thought += (
                    f"--- Short-term Memory {index} ---\n"
                    f"Received: {short_term.receive_context}\n"
                    f"Sent: '{short_term.context}' to '{short_term.objective}'\n"
                )
        if self.use_rag:
            des_thought += f"**RAG Knowledge:**\n{self.long_memory}\n"

        return AgentMessage(agent_message.receive, agent_message.send, des_thought)

    def add_conversation(self, des_thought, src_thought) -> AgentMessage:
        if src_thought.prompt != "":
            des_thought.prompt += f"\n**Received:** {src_thought.prompt}\nFrom Agent {des_thought.receive}.\n"
        des_thought.prompt += THINK_TEMPLATE
        return des_thought

    async def _generate(self, prompt: str) -> str:
        if not self.aclient: return "{}"
        try:
            completion = await self.aclient.chat.completions.create(
                model=self.model["name"],
                messages=[{"role": "user", "content": prompt}]
            )
            return str(completion.choices[0].message.content)
        except Exception:
            time.sleep(2)
            return "{}"
    
    def _save_to_rag(self, log: Log) -> None:
        if not self.embedding: return
        try:
            main_db = FAISS.load_local(self.rag_dir, self.embedding, allow_dangerous_deserialization=True)
            new_memory = [Document(page_content=log.context, metadata={"timestamp": log.timestamp})]
            main_db.add_documents(new_memory)
            main_db.save_local(self.rag_dir)
        except: pass

    def _search_rag(self, query: str, num: int) -> list[Document]:
        if not self.embedding: return []
        try:
            db = FAISS.load_local(self.rag_dir, self.embedding, allow_dangerous_deserialization=True)
            return db.similarity_search(query, k=num)
        except: return []

    def _memorize(self, log: Log) -> None:
        if len(self.short_term_memory) < self.max_memory:
            self.short_term_memory.append(log)
        else:
            mem = self.short_term_memory.pop(0)
            if self.use_rag: self._save_to_rag(mem)
            self.short_term_memory.append(log)

    def receive_information(self) -> AgentMessage:
        self.received_messages = ""
        if len(self.message_buffer):
            self.conversation_buffer = self.message_buffer.pop(0)
            self.received_messages = self.conversation_buffer.prompt
            text_to_consider = self.add_background(self.conversation_buffer, self.background, self.short_term_memory, self.rag_dir)
            text_to_consider = self.add_conversation(text_to_consider, self.conversation_buffer)
        else:
            if self.background.actively_chat_probability > random.random():
                receive_nb = random.choice(self.background.neighbors) if self.background.neighbors else self.name
                self.conversation_buffer = AgentMessage(receive_nb, self.name, "")
                text_to_consider = self.add_background(self.conversation_buffer, self.background, self.short_term_memory, self.rag_dir)
                text_to_consider = self.add_conversation(text_to_consider, self.conversation_buffer)
            else:
                return AgentMessage(-1, -1, "<waiting>")
        return text_to_consider

    async def _think(self, text_to_consider: AgentMessage, print_prompt: bool, print_log: bool) -> Action:
        """
        Processes the input context and generates an Action using the LLM.
        Includes retry logic for robust JSON generation.
        """
        max_retries = 5 
        item = {}
        
        for attempt in range(max_retries):
            raw_result = await self._generate(text_to_consider.prompt)
            raw_result_clean = clean_json_string(raw_result)
            
            try:
                item = json.loads(raw_result_clean)
                break 
            except json.JSONDecodeError:
                time.sleep(1)
        
        if not item:
            item = {"type": "send_message", "reply_prompt": "(System: Failed to generate response)", "sending_target": []}

        if print_log: print(raw_result)

        target = item.get("sending_target", [])
        action_type = item.get("type", "unknown_type") 
        tool_name = item.get("tool_name", None)
        reply_prompt = item.get("reply_prompt", "")

        return Action(action_type, tool_name, reply_prompt, target)

    async def _act(
            self,
            action: Action,
            text_to_consider: AgentMessage,
            correct_end: List[int] = None,
            defense = None,
            print_prompt: bool = False,
            print_log: bool = False,
    ):
        """
        Executes the determined action (Use Tool or Send Message).
        """
        send_target = []
        temp_log = []
        message_log = []

        tool_calls = 0
        MAX_TOOL_CALLS = 3

        # Handle tool usage loop
        while action.type == 'use_tool' and tool_calls < MAX_TOOL_CALLS:
            tool_calls += 1
            if self.hijack_tool:
                tools_output = await call_hijack_tool(
                    action.tool_name,
                    action.reply_prompt,
                    self.model,
                    self.aclient,
                    self.extra_data
                )
            else:
                tools_output = await call_tool(
                    action.tool_name,
                    action.reply_prompt,
                    self.model,
                    self.aclient,
                )

            prompt = (
                f"\n\n**Tool Execution Result**\n\n"
                f"You previously invoked the tool: **{action.tool_name}**\n\n"
                f"The tool has returned the following output:\n\n"
                f"{tools_output}\n\n"
                f"**Next Step:**\n"
                f"Now, incorporating this tool output into your reasoning, please continue working on your subtask. Decide your next action "
                f"(e.g., `send_message` to report findings/ask questions, or `use_tool` again if necessary) and format your response according to the action output specifications.\n"
            )

            action.reply_prompt += prompt
            log = Log(self.id, self.name, None, action.type, action.reply_prompt, self.received_messages)
            try:
                async with aiofiles.open(self.log_path, mode='a', encoding='utf-8') as afp:
                    await afp.write(json.dumps(log.convert_to_json()) + "\n")
            except Exception:
                pass

            temp_log.append(log.convert_to_json())
            self._memorize(log)
            text_to_consider.prompt += f"\n{action.reply_prompt}"

            action = await self._think(text_to_consider, False, False)

            if not action.sending_target:
                break
            if set(action.sending_target).issubset(self.background.neighbors):
                break

        if action.type == 'send_message':
            raw_targets = action.sending_target or []
            valid_targets = []
            
            for t in raw_targets:
                try:
                    t_int = int(t)
                except Exception:
                    continue
                if t_int == self.name:
                    continue
                if 0 <= t_int < len(self.simulator.agents):
                    valid_targets.append(t_int)

            send_target = valid_targets

            for i in send_target:
                result = AgentMessage(self.name, i, action.reply_prompt)
                if print_log:
                    print(result.prompt)

                # Apply TopoSHIELD defense intervention if active
                if correct_end is not None and i in correct_end and defense:
                    result = await defense.correct_cot(result)

                self.simulator.agents[i].message_buffer.append(result)

                t_log = Log(self.id, result.send, result.receive, action.type, result.prompt, self.received_messages)
                try:
                    async with aiofiles.open(self.log_path, mode='a', encoding='utf-8') as afp:
                        await afp.write(json.dumps(t_log.convert_to_json()) + "\n")
                except Exception:
                    pass

            log = Log(self.id, self.name, send_target, action.type, action.reply_prompt, self.received_messages)
            temp_log.append(log.convert_to_json())
            message_log.append(log.convert_to_json())
            self._memorize(log)
            return temp_log, message_log

        return [], []

    async def emulate_one_step(self, print_log=False, print_prompt=False, correct_end=None, defense=None):
        text_to_consider = self.receive_information()
        if text_to_consider.prompt == "<waiting>": return [], []
        actions = await self._think(text_to_consider, print_prompt, print_log)
        return await self._act(actions, text_to_consider, correct_end, defense, print_prompt, print_log)


class PlanningAgent(Agent):
    """
    Agent responsible for initial task planning and topology determination.
    """

    def __init__(self, id, name, model, tools, background, simulator, use_rag, max_memory=5, log_path=None):
        super().__init__(
            id=id,
            name=name,
            model=model,
            tools=tools,
            background=background,
            simulator=simulator,
            use_rag=use_rag,
            extra_data={},
            max_memory=max_memory,
            log_path=log_path
        )
        self.messages = []

        try:
            model["base_url"] = _normalize_base_url(model.get("base_url"))
        except Exception:
            pass

        self.client = OpenAI(api_key=model['api_key'], base_url=model.get('base_url'))

    def _generate(self, input: str, model):
        self.messages.append({"role": "user", "content": input})
        try:
            generated_text = self.client.chat.completions.create(model=model['name'], messages=self.messages)
            self.messages.append(generated_text.choices[0].message)
            return str(generated_text.choices[0].message.content)
        except: return ""

    def _rethink(self):
        text_to_consider = "Generate the JSON assignment list now. Don't generate ```!!!\n"
        for _ in range(5):
            try:
                raw_subtask = self._generate(text_to_consider, self.model)
                raw_subtask = clean_json_string(raw_subtask)
                return json.loads(raw_subtask)
            except: time.sleep(1)
        return []

    def determine_topo(self):
        text_to_consider = "Generate the JSON topology list now:\n"
        for _ in range(5):
            try:
                raw_topo = self._generate(text_to_consider, self.model)
                raw_topo = clean_json_string(raw_topo)
                return json.loads(raw_topo)
            except: time.sleep(1)
        return []

    def emulate_one_step(self, self_deter: bool, print_log=False, print_prompt=False):
        mid_plan = self._generate(self.background.info, self.model)
        planning = self._rethink()
        if self_deter:
            topo = self.determine_topo()
            return planning, topo
        return planning, None

class ConclusionAgent(Agent):
    """
    Agent responsible for synthesizing the final conclusion.
    """
    def __init__(self, id, name, model, tools, background, simulator, use_rag, max_memory = 5, log_path=None):
        super().__init__(id, name, model, tools, background, simulator, use_rag, max_memory, log_path=log_path)

    async def emulate_one_step(self, print_log=False, print_prompt=False) -> str:
        conclusion = await self._generate(self.background.info)
        log = ConclusionLog(self.id, context=conclusion)
        try:
            async with aiofiles.open(self.log_path, mode='a', encoding='utf-8') as afp:
                await afp.write(json.dumps(log.convert_to_json()) + '\n')
        except: pass
        return log.convert_to_str()

class HijackAgent(Agent):
    """
    Adversarial agent designed to inject misinformation.
    """
    def __init__(self, id, name, model, tools, background, simulator, use_rag, extra_data, log_path):
        super().__init__(id, name, model, tools, background, simulator, use_rag, extra_data, log_path=log_path)

    async def _think(self, text_to_consider: AgentMessage, print_prompt: bool, print_log: bool) -> Action:
        text_to_consider.prompt += f"Attention: ... {self.extra_data['misinfo_goal']} ..."
        for i in self.extra_data["misinfo_argument"]: text_to_consider.prompt += i
        
        max_retries = 5
        item = {}
        for _ in range(max_retries):
            raw_result = await self._generate(text_to_consider.prompt)
            raw_result = clean_json_string(raw_result)
            try:
                item = json.loads(raw_result)
                break
            except: time.sleep(1)
        
        if not item: item = {"type": "send_message", "sending_target": [], "reply_prompt": "Error"}

        if print_log: print(raw_result)
        target = item.get("sending_target", [])
        action_type = item.get("type", "unknown_type")
        tool_name = item.get("tool_name", None)
        reply_prompt = item.get("reply_prompt", "")
        return Action(action_type, tool_name, reply_prompt, target)

class DefenceAgent:
    """
    Helper agent class utilized by the TopoSHIELD mechanism to analyze and correct messages.
    """
    possible_goal = []
    possible_raw_goal = []
    _list_lock = asyncio.Lock()

    def __init__(self, ground_truth=None, model_config=None):
        self.ground_truth = ground_truth
        self.messages = []
        from sandbox.prompt import COT_PROMPT
        self.cot_prompt = COT_PROMPT

        if model_config:
            try:
                model_config["base_url"] = _normalize_base_url(model_config.get("base_url"))
            except Exception:
                pass

            self.client = OpenAI(api_key=model_config['api_key'], base_url=model_config['base_url'])
            self.model_name = model_config['name']
        else:
            self.client = None
            self.model_name = None

    def _generate(self, input_str):
        if not self.client: return ""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input_str}],
                timeout=60
            )
            return str(completion.choices[0].message.content)
        except: return ""

    async def correct_cot(self, input_msg):
        """
        Analyzes the input message for potential misinformation and generates a correction if necessary.
        """
        original_input = input_msg.prompt
        cot_template = self.cot_prompt + f"\nInput: {original_input}\n..."
        
        raw = self._generate(cot_template)
        
        try:
            if raw:
                text = raw.strip()
                if text.startswith("```json"): text = text[7:]
                if text.endswith("```"): text = text[:-3]
                text = text.strip()
                
                res = json.loads(text)
                if "revised_message" in res:
                    input_msg.prompt = res["revised_message"]
        except:
            pass
            
        return input_msg

# --- END OF FILE sandbox/agent.py ---