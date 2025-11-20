"""
FinVault - Streamlit Financial Analysis Application
====================================================
This is the user interface (UI) for the multi-agent financial analysis system.
It uses Streamlit to create an interactive web application where users can:
1. Chat with AI agents to analyze stocks and companies
2. See agent reasoning in real-time (supervisor, analysts, synthesizer)
3. Get structured financial analysis with recommendations

Tech Stack:
- Streamlit: Web UI framework
- LangGraph: Multi-agent orchestration (from agent.py)
- LangChain: LLM integration
- SEC EDGAR: Financial data access
"""

import uuid

import streamlit as st
from dotenv import load_dotenv
from edgar import set_identity
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage

from agent import AgentName, AgentState, ContextSchema, create_agent
from config import Config, ModelProvider

# INITIALIZATION & CONFIGURATION

# Load Environment Variables

# Loads variables from .env file (if present)
# Common variables might include:
# - API keys (OpenAI, Anthropic, etc.)
# - Database credentials
# - Custom configuration overrides

load_dotenv()

# Streamlit Page Configuration

# Must be called FIRST before any other Streamlit commands
# Sets browser tab title and layout mode

st.set_page_config(
    page_title="FinVault",  # Shows in browser tab
    layout="wide"           # Uses full browser width (vs "centered")
)

# Page Header

# :material/candlestick_chart: is a Streamlit icon (material design)
# Creates the main title and tagline for the app

st.header(":material/candlestick_chart: FinVault")
st.subheader("Private AI assistant for financial analysis")

# CACHED RESOURCE FUNCTIONS

# @st.cache_resource decorators ensure these expensive operations run only once
# and are shared across all user sessions
#
# Why cache?
# - Workflow and model initialization is expensive (takes seconds)
# - Without caching, it would run on every page refresh
# - With caching, runs once and reuses the result
#
# When cache is cleared?
# - Manual: User clicks "Clear cache" in Streamlit menu
# - Automatic: Code changes detected, app restarts

@st.cache_resource
def create_workflow():
    """
    Creates and caches the LangGraph agent workflow.
    
    This is the multi-agent system from agent.py:
    - Supervisor (orchestrator)
    - Price Analyst (stock prices)
    - Filing Analyst (SEC filings)
    - Synthesizer (final answer)
    
    Cached because:
    - Graph compilation is expensive
    - Same workflow used for all queries
    - No need to recreate per user or per query
    
    Returns:
        Compiled LangGraph workflow ready to execute
    """
    return create_agent()

@st.cache_resource
def create_model():
    """
    Creates and caches the LLM (Language Model) instance.
    
    Configuration:
    - Model name and provider from Config (e.g., "qwen3:8b" via Ollama)
    - Temperature for response randomness
    - Context window size (max tokens)
    
    Why cache?
    - Model initialization involves loading configurations
    - Same model used for all agents and all users
    - Prevents redundant setup on every query
    
    Returns:
        Initialized LangChain chat model
    """

    # Build Model Parameters

    parameters = {
        "temperature": Config.MODEL.temperature,  # 0.0 = deterministic
        "thinking_budget": 0,  # For models with extended thinking (like o1)
                                # 0 = no extended thinking
    }

    # Provider-Specific Parameters

    # Ollama requires explicit context window setting
    # Other providers (OpenAI, Anthropic) use defaults

    if Config.MODEL.provider == ModelProvider.OLLAMA:
        parameters["num_ctx"] = Config.CONTEXT_WINDOW  # Max tokens: 8192

    # Initialize Model

    # Format: "provider:model_name" (e.g., "ollama:qwen3:8b")
    # init_chat_model is LangChain's universal model initializer

    return init_chat_model(
        f"{Config.MODEL.provider.value}:{Config.MODEL.name}",
        **parameters,
    )

# SESSION STATE INITIALIZATION

# Streamlit session state persists data across page reruns within a user session
# Think of it as a dictionary that survives the script re-executing
#
# Session State Structure:
#   st.session_state = {
#       "messages": [...],      ← Conversation history
#       "thread_id": "uuid",    ← Unique conversation identifier
#       "email": "user@..."     ← User's email (for SEC EDGAR access)
#   }
#
# Why needed?
# - Streamlit reruns the entire script on every interaction
# - Without session state, variables would reset to initial values
# - Session state preserves conversation history, user email, etc.

if "messages" not in st.session_state:
    st.session_state.messages = []  # List of HumanMessage & AIMessage objects

if "thread_id" not in st.session_state:
    # Unique ID for this conversation thread
    # Could be used for logging, analytics, or database storage
    st.session_state.thread_id = str(uuid.uuid4())

if "email" not in st.session_state:
    st.session_state.email = None  # Will be set after user provides email

# EMAIL COLLECTION GATE

# SEC EDGAR requires user identification for API access (to prevent abuse)
# This section blocks app usage until a valid email is provided
#
# Flow:
#   1. Check if email exists in session state
#   2. If not → Show email input form
#   3. Validate email format
#   4. Set identity in EDGAR library
#   5. Store in session state and reload
#   6. If yes → Allow app usage

if not st.session_state.email:

    # Show Email Collection UI

    st.info("Welcome! To get started, please enter your email address below.")
    
    email_input = st.text_input(
        "Your Email Address",
        placeholder="your.email@company.com",
        help="Required to access SEC EDGAR filings.",  # Tooltip on hover
    )

    if email_input:

        # Validate Email Format (Basic)

        # Checks:
        # 1. Contains "@" symbol
        # 2. Has "." after "@" (e.g., "@gmail.com" is valid, "@gmail" is not)

        if "@" in email_input and "." in email_input.split("@")[1]:
            # Valid email format
            st.session_state.email = email_input
            
            # Set EDGAR Identity

            # SEC requires identification for rate limiting and compliance
            # This tells the edgar library who is making requests

            set_identity(email_input)
            
            st.success("Identity was Set! Reloading...")
            
            # Rerun Script to Refresh UI

            # Forces Streamlit to re-execute the entire script
            # On rerun, email will be in session state, so gate will pass

            st.rerun()
        else:
            # Invalid email format
            st.error("Please enter a valid email address.")
    
    # Stop Script Execution

    # st.stop() prevents code below this point from running
    # User is stuck at email gate until valid email is provided

    st.stop()
else:
    # Email Already Collected - Set Identity

    # Even if email is in session state, we need to set_identity on each run
    # because the EDGAR library doesn't persist across script reruns

    set_identity(st.session_state.email)

# SIDEBAR - CONVERSATION CONTROLS

# Streamlit's sidebar provides a persistent left panel for controls
# 
# Visual Layout:
#   ┌────────────┬──────────────────────────────┐
#   │  SIDEBAR   │   MAIN CONTENT AREA          │
#   │            │                              │
#   │ [New Conv] │   Chat messages              │
#   │            │   Input box                  │
#   │ Messages:5 │                              │
#   └────────────┴──────────────────────────────┘

with st.sidebar:
    st.header("Conversation")
    
    # New Conversation Button

    # Allows user to start fresh (clear message history)
    # use_container_width=True makes button full width of sidebar

    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []  # Clear conversation history
        st.rerun()  # Refresh UI to show empty chat

    # Message Counter

    # Shows how many messages are in current conversation
    # Only displays if messages exist (avoids "Messages: 0")
    # st.caption() creates small gray text

    if st.session_state.messages:
        st.caption(f"Messages in conversation: {len(st.session_state.messages)}")

# HELPER FUNCTION - MARKDOWN ESCAPING

# Why needed?
# - Streamlit renders markdown by default
# - Dollar signs ($) trigger LaTeX math mode in markdown
# - Financial data has lots of dollar signs: "$500M revenue"
# - Without escaping: "$500M" might render as math, causing errors
# - With escaping: "\$500M" displays correctly as text

def escape_markdown(text: str) -> str:
    """
    Escapes special markdown characters to prevent rendering issues.
    
    Currently only escapes dollar signs, but could be extended to handle:
    - Asterisks (*) for italics/bold
    - Underscores (_) for emphasis
    - Brackets ([]) for links
    
    Args:
        text: Raw text that might contain special characters
        
    Returns:
        Text with special characters escaped for safe markdown rendering
    """
    return text.replace("$", r"\$")  # r"\$" is a raw string with backslash

# DISPLAY CONVERSATION HISTORY

# Renders all previous messages in the conversation
# Distinguishes between user messages and AI responses with different avatars
#
# Message Display:
#   👤 User: What's NVDA's stock price?
#   🤖 Assistant: Let me analyze that for you...

for messages in st.session_state.messages:
    avatar = None
    
    # Determine Message Type and Avatar

    if isinstance(messages, HumanMessage):
        avatar = ":material/person:"  # Person icon for user messages
        role = "user"
    elif isinstance(messages, AIMessage):
        avatar = ":material/smart_toy:"  # Robot icon for AI messages
        role = "assistant"
    else:
        # Skip other message types (ToolMessage, SystemMessage, etc.)
        # These are internal and not meant for display
        continue

    # Render Message in Chat Bubble

    # st.chat_message() creates a styled message container
    # Automatically applies avatar and role-based styling

    with st.chat_message(role, avatar=avatar):
        st.markdown(escape_markdown(messages.content))

# CHAT INPUT & MESSAGE PROCESSING (MAIN INTERACTION LOOP)

# This is the core of the application - where user queries are processed
#
# Flow:
#   1. User types query and presses Enter
#   2. Query is added to conversation history
#   3. Agent workflow is invoked
#   4. Agent reasoning is displayed in real-time (streaming)
#   5. Final answer is shown to user
#   6. Conversation history is updated

# Chat Input Widget

# st.chat_input() creates the text input box at the bottom of the page
# Returns user input when Enter is pressed, None otherwise
# The := (walrus operator) assigns and checks in one line

if prompt := st.chat_input(
    "Ask about company filings and stock prices... (e.g., 'Analyze Tesla stock')"
):
    # STEP 1: Add User Message to History

    st.session_state.messages.append(HumanMessage(content=prompt))

    # STEP 2: Display User's Message

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    # STEP 3: Prepare Agent State

    # Create AgentState with copy of messages
    # .copy() ensures modifications during agent execution don't affect
    # the original session state until we're done

    initial_state: AgentState = AgentState(
        messages=st.session_state.messages.copy()
    )

    # STEP 4: Create Assistant Message Container

    # Everything below renders in the assistant's chat bubble

    with st.chat_message("assistant", avatar=":material/smart_toy:"):

        # Create Status Container (Expandable Section)

        # Shows agent reasoning process (Supervisor, Analysts)
        # expanded=True shows content by default
        # User can collapse it to hide the "thinking" process
 
        status_container = st.status(
            "**Analyzing your request...**", 
            expanded=True
        )

        # Create Placeholder for Final Answer

        # st.empty() creates a placeholder that can be updated
        # Will show the Synthesizer's output (final answer)

        final_answer_placeholder = st.empty()

        # Initialize Tracking Variables

        current_agent = None  # Track which agent is currently speaking
        agent_containers = {}  # Map agent names to their UI containers
        agent_content = {}     # Map agent names to their accumulated content

        # STEP 5: Get Workflow Instance

        workflow = create_workflow()  # Cached, so no recompilation

        # STEP 6: STREAM AGENT EXECUTION (REAL-TIME UPDATES)

        # workflow.stream() yields messages as they're generated
        # This enables real-time UI updates (like ChatGPT's streaming)
        #
        # Stream Structure:
        #   for message_chunk, metadata in workflow.stream(...):
        #       message_chunk = Piece of content (text chunk)
        #       metadata = {"langraph_node": "Supervisor", ...}
        #
        # Why stream mode "messages"?
        # - Alternative: stream_mode="values" (complete state after each step)
        # - "messages" gives us individual message chunks as they're created
        # - Better for real-time display and user experience

        for message_chunk, metadata in workflow.stream(
            initial_state,
            context=ContextSchema(model=create_model()),  # Cached model
            stream_mode="messages"  # Stream individual message chunks
        ):

            # Extract Agent Name from Metadata

            # Metadata tells us which node (agent) generated this chunk

            agent_name = metadata.get("langgraph_node", "Unknown")

            # Skip Empty Chunks

            # Some chunks might be control signals with no content

            if not hasattr(message_chunk, "content") or not message_chunk.content:
                continue

            content = str(message_chunk.content)

            # Handle Agent Switch (New Agent Started Speaking)

            if agent_name != current_agent:
                current_agent = agent_name
                agent_content[agent_name] = ""  # Reset content accumulator

                # Create UI Container for New Agent

                with status_container:
                    if agent_name in [
                        AgentName.SUPERVISOR,
                        AgentName.PRICE_ANALYST,
                        AgentName.FILING_ANALYST
                    ]:
                        # Show agent name as header in status container
                        st.markdown(f"**{agent_name}**")
                        # Create empty placeholder for this agent's content
                        agent_containers[agent_name] = st.empty()
                        
                    elif agent_name == AgentName.SYNTHESIZER:
                        # Synthesizer gets special treatment
                        # Update status label to "Synthesizing..."
                        status_container.update(
                            label="**Synthesizing final answer...**"
                        )
                        agent_containers[agent_name] = st.empty()

            # Accumulate Content

            # Each chunk is a piece of the agent's response
            # Accumulate chunks to show complete message

            agent_content[agent_name] += content

            # Update UI with Accumulated Content

            if agent_name in agent_containers:
                if agent_name == AgentName.SYNTHESIZER:

                    # Synthesizer Output → Final Answer Area

                    # Display in main chat area (outside status container)
                    # This is what the user ultimately sees as the answer

                    final_answer_placeholder.markdown(
                        "### Analysis\n" + escape_markdown(agent_content[agent_name])
                    )
                else:

                    # Other Agents → Status Container

                    # Show reasoning process in collapsible status section

                    with status_container:
                        if agent_name == AgentName.SUPERVISOR:
                            # Supervisor gets info box (blue background)
                            agent_containers[agent_name].info(
                                escape_markdown(agent_content[agent_name])
                            )
                        else:
                            # Workers get regular markdown (no background)
                            agent_containers[agent_name].markdown(
                                escape_markdown(agent_content[agent_name])
                            )

        # STEP 7: Mark Status Container as Complete

        # After all agents finish, update the status container:
        # - Label changes to "Thoughts"
        # - State changes to "complete" (shows green checkmark)
        # - expanded=False collapses it (hides reasoning by default)

        status_container.update(
            label="**Thoughts**",
            state="complete",  # "complete", "running", or "error"
            expanded=False     # Collapse to focus on final answer
        )

# END OF SCRIPT

# After processing completes:
# 1. Final answer is displayed to user
# 2. Conversation history is preserved in session state
# 3. User can ask follow-up questions
# 4. Script reruns on each interaction, but session state persists

# USAGE & DEPLOYMENT GUIDE


# Running the Application:
# ────────────────────────

# 1. Local Development:
#    streamlit run app.py

# 2. With Custom Port:
#    streamlit run app.py --server.port 8080

# 3. Accessible on Network:
#    streamlit run app.py --server.address 0.0.0.0


# Application Flow (Visual):
# ──────────────────────────

# ┌─────────────────────────────────────────────────────────────────────┐
# │ 1. USER ENTERS EMAIL                                                │
# │    └─→ Validates format                                             │
# │        └─→ Sets EDGAR identity                                      │
# │            └─→ Enables SEC filing access                            │
# └─────────────────────────────────────────────────────────────────────┘
#                               ↓
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 2. USER TYPES QUERY: "Analyze NVDA stock"                           │
# │    └─→ Query added to session_state.messages                        │
# │        └─→ AgentState created with message history                  │
# └─────────────────────────────────────────────────────────────────────┘
#                               ↓
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 3. AGENT WORKFLOW STREAMS EXECUTION                                 │
# │    ┌──────────────────────────────────────────────────┐             │
# │    │ Supervisor: "Need price and filing data"         │             │
# │    │   → Displayed in status container (blue box)     │             │
# │    └──────────────────────────────────────────────────┘             │
# │                         ↓                                           │
# │    ┌──────────────────────────────────────────────────┐             │
# │    │ Price Analyst: "NVDA up 35% in 90 days..."       │             │
# │    │   → Displayed in status container (markdown)     │             │
# │    └──────────────────────────────────────────────────┘             │
# │                         ↓                                           │
# │    ┌──────────────────────────────────────────────────┐             │
# │    │ Filing Analyst: "Key risks: supply chain..."     │             │
# │    │   → Displayed in status container (markdown)     │             │
# │    └──────────────────────────────────────────────────┘             │
# │                         ↓                                           │
# │    ┌──────────────────────────────────────────────────┐             │
# │    │ Synthesizer: "### Analysis                       │             │
# │    │ **Overview**: NVIDIA is a leading...             │             │
# │    │ **Price Action**: Strong uptrend...              │             │
# │    │ **Recommendation**: BUY"                         │             │
# │    │   → Displayed in main chat area                  │             │
# │    └──────────────────────────────────────────────────┘             │
# └─────────────────────────────────────────────────────────────────────┘
#                               ↓
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 4. STATUS CONTAINER COLLAPSES                                       │
# │    - Label: "Thoughts" (instead of "Analyzing...")                  │
# │    - State: Complete (green checkmark)                              │
# │    - Expanded: False (collapsed by default)                         │
# │    User can click to expand and see agent reasoning                 │
# └─────────────────────────────────────────────────────────────────────┘
#                               ↓
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 5. USER CAN ASK FOLLOW-UP QUESTIONS                                 │
# │    - Full conversation history is preserved                         │
# │    - Agents have context from previous exchanges                    │
# │    - OR click "New Conversation" to start fresh                     │
# └─────────────────────────────────────────────────────────────────────┘


# Streamlit-Specific Concepts Explained:
# ──────────────────────────────────────

# 1. SCRIPT RERUN MODEL:
#    ─────────────────────
#    Every interaction (button click, input, etc.) causes the ENTIRE script
#    to re-execute from top to bottom.
   
#    Example:
#    - User types message → Script reruns → Processes message → Displays result
#    - User clicks button → Script reruns → Button handler runs
   
#    Why this matters:
#    - Variables defined without session state will reset to initial values
#    - @st.cache_resource ensures expensive operations don't repeat
#    - Must use session_state to preserve data across reruns

# 2. SESSION STATE:
#    ───────────────
#    Dictionary-like object that persists data across reruns WITHIN a session.
   
#    Session = Single user's browser tab/window
#    - Different users have separate session states
#    - Closing tab/browser clears session state
#    - Refreshing page creates new session
   
#    Access pattern:
#    ```python
#    # Check if key exists
#    if "key" not in st.session_state:
#        st.session_state.key = initial_value
   
#    # Read value
#    value = st.session_state.key
   
#    # Update value
#    st.session_state.key = new_value
#    ```

# 3. CACHING (@st.cache_resource vs @st.cache_data):
#    ─────────────────────────────────────────────────
#    @st.cache_resource:
#    - For singletons (database connections, ML models, workflows)
#    - Shared across ALL users and ALL sessions
#    - Not thread-safe by default (use for immutable objects)
#    - Example: LangGraph workflow, LLM model
   
#    @st.cache_data:
#    - For expensive computations that return data
#    - Creates separate copy per call (thread-safe)
#    - Example: Loading CSV, API calls, data transformations
   
#    When to use which:
#    - Model/connection → cache_resource
#    - Data processing → cache_data

# 4. STREAMLIT CONTAINERS:
#    ──────────────────────
#    Containers let you organize and update specific UI sections.
   
#    st.empty():
#    - Creates placeholder that can be updated
#    - Each update REPLACES previous content
#    - Example: final_answer_placeholder.markdown("New text")
   
#    st.status():
#    - Creates expandable section with label and state
#    - States: "running", "complete", "error"
#    - Can contain multiple sub-elements
   
#    st.chat_message():
#    - Creates styled chat bubble with avatar
#    - Automatically handles user/assistant styling
#    - Uses with block: all content inside appears in bubble

# 5. STREAMING PATTERN:
#    ───────────────────
#    The stream loop in this app demonstrates real-time UI updates:
   
#    ```python
#    for chunk, metadata in workflow.stream(...):
#        agent_content[agent] += chunk  # Accumulate
#        container.markdown(agent_content[agent])  # Update UI
#    ```
   
#    Why accumulate instead of just displaying chunks?
#    - Chunks might be partial words or tokens
#    - Accumulating shows complete, readable text
#    - Each update replaces previous (so no duplicate text)


# Key Design Decisions:
# ────────────────────

# 1. WHY COPY MESSAGES FOR AGENT STATE?
#    ```python
#    initial_state = AgentState(messages=st.session_state.messages.copy())
#    ```
   
#    Reason:
#    - Agent execution might modify the message list internally
#    - Don't want those modifications affecting session state mid-execution
#    - After agent completes, final message is added to session state
   
#    Alternative (problematic):
#    ```python
#    # DON'T DO THIS - agents modify session state directly
#    initial_state = AgentState(messages=st.session_state.messages)
#    ```

# 2. WHY SEPARATE STATUS CONTAINER AND FINAL ANSWER?
   
#    Status Container:
#    - Shows agent "thinking" process
#    - Supervisor delegations, worker summaries
#    - Can be collapsed to hide details
   
#    Final Answer:
#    - Shows synthesizer's polished response
#    - What user actually wants to read
#    - Always visible, not collapsible
   
#    Benefit:
#    - Advanced users can see reasoning
#    - Regular users get clean answer
#    - Best of both worlds

# 3. WHY ESCAPE MARKDOWN FOR DOLLAR SIGNS?
   
#    Problem:
#    - Financial data: "Company earned $500M revenue"
#    - Streamlit markdown: "$" triggers LaTeX math mode
#    - Result: "$500M" might break or render as math equation
   
#    Solution:
#    - escape_markdown() converts "$" → "\$"
#    - Displays as regular text, not math
   
#    Example:
#    ```python
#    # Without escaping
#    st.markdown("Revenue: $500M")  # Might break
   
#    # With escaping
#    st.markdown("Revenue: \$500M")  # Works correctly
#    ```

# 4. WHY CHECK EMAIL FORMAT ON FRONTEND?
   
#    Basic validation:
#    - Checks for "@" and "." after domain
#    - Catches obvious typos (user@gmailcom)
   
#    Not comprehensive:
#    - Doesn't verify email actually exists
#    - Doesn't check against email RFC standards
#    - Just prevents common mistakes
   
#    SEC requirement:
#    - EDGAR needs identification, not verification
#    - Any valid-looking email works for API access


# Common Streamlit Patterns Used:
# ───────────────────────────────

# 1. CONDITIONAL UI RENDERING:
#    ```python
#    if condition:
#        st.info("Message")
#        st.stop()  # Prevent rest of script from running
#    ```

# 2. SIDEBAR PATTERN:
#    ```python
#    with st.sidebar:
#        # All UI elements here appear in sidebar
#        st.button("...")
#    ```

# 3. CHAT INTERFACE PATTERN:
#    ```python
#    # Display history
#    for msg in messages:
#        with st.chat_message(role):
#            st.markdown(msg.content)
   
#    # Get new input
#    if prompt := st.chat_input("..."):
#        # Process and display response
#    ```

# 4. STREAMING UPDATES:
#    ```python
#    placeholder = st.empty()
#    for chunk in stream:
#        accumulated_text += chunk
#        placeholder.markdown(accumulated_text)
#    ```


# Debugging Tips:
# ───────────────

# 1. VIEW SESSION STATE:
#    ```python
#    with st.sidebar:
#        st.write("Debug - Session State:")
#        st.write(st.session_state)
#    ```

# 2. LOG AGENT EXECUTION:
#    ```python
#    for chunk, metadata in workflow.stream(...):
#        st.sidebar.write(f"Agent: {metadata.get('langraph_node')}")
#        st.sidebar.write(f"Content: {chunk.content[:50]}")
#    ```

# 3. CHECK MESSAGE TYPES:
#    ```python
#    for msg in st.session_state.messages:
#        st.sidebar.write(f"{type(msg).__name__}: {msg.content[:30]}")
#    ```

# 4. FORCE CACHE CLEAR:
#    - Streamlit menu (three dots) → "Clear cache"
#    - Or restart app: Ctrl+C in terminal, then rerun

# 5. VIEW EXCEPTIONS:
#    Streamlit shows exceptions in UI automatically
#    Add try-except for custom error messages:
#    ```python
#    try:
#        result = workflow.invoke(...)
#    except Exception as e:
#        st.error(f"Error: {str(e)}")
#        st.exception(e)  # Shows full traceback
#    ```


# Deployment Considerations:
# ─────────────────────────

# 1. ENVIRONMENT VARIABLES:
#    Create .env file:
#    ```
#    OPENAI_API_KEY=sk-...
#    ANTHROPIC_API_KEY=sk-ant-...
#    APP_HOME=/path/to/app
#    ```

# 2. REQUIREMENTS.TXT:
#    ```
#    streamlit
#    langchain
#    langchain-ollama  # or langchain-openai, etc.
#    langgraph
#    yfinance
#    edgartools
#    python-dotenv
#    pydantic
#    ```

# 3. RUN IN PRODUCTION:
#    ```bash
#    # Basic
#    streamlit run app.py
   
#    # With custom config
#    streamlit run app.py \
#      --server.port 8080 \
#      --server.address 0.0.0.0 \
#      --server.headless true
#    ```

# 4. DOCKER DEPLOYMENT:
#    ```dockerfile
#    FROM python:3.11-slim
#    WORKDIR /app
#    COPY requirements.txt .
#    RUN pip install -r requirements.txt
#    COPY . .
#    EXPOSE 8501
#    CMD ["streamlit", "run", "app.py"]
#    ```

# 5. CLOUD PLATFORMS:
#    - Streamlit Cloud (streamlit.io/cloud): Native platform, easiest
#    - Heroku: Add Procfile with streamlit run command
#    - AWS/GCP/Azure: Docker container or VM with exposed port


# Performance Optimization:
# ────────────────────────

# 1. CACHE EVERYTHING EXPENSIVE:
#    ```python
#    @st.cache_resource
#    def load_model():
#        return ChatOllama(...)
   
#    @st.cache_data(ttl=3600)  # Cache for 1 hour
#    def fetch_market_data():
#        return yf.download(...)
#    ```

# 2. LAZY LOADING:
#    Only load resources when needed:
#    ```python
#    if "model" not in st.session_state:
#        st.session_state.model = create_model()
#    ```

# 3. STREAM MODE FOR RESPONSIVENESS:
#    Using stream_mode="messages" provides instant feedback
#    Alternative stream_mode="values" waits for complete steps

# 4. LIMIT CONVERSATION HISTORY:
#    ```python
#    MAX_MESSAGES = 20
#    if len(st.session_state.messages) > MAX_MESSAGES:
#        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]
#    ```


# Security Considerations:
# ───────────────────────

# 1. EMAIL COLLECTION:
#    - Currently stored only in session (not persistent)
#    - Consider encrypting if storing in database
#    - Don't share emails between users

# 2. API KEYS:
#    - Always use .env file, never hardcode
#    - Add .env to .gitignore
#    - Use secrets management in production:
#      ```python
#      import streamlit as st
#      api_key = st.secrets["OPENAI_API_KEY"]
#      ```

# 3. USER INPUT VALIDATION:
#    - Validate ticker symbols before API calls
#    - Limit query length to prevent abuse
#    - Rate limiting for production deployments

# 4. ERROR HANDLING:
#    - Don't expose internal errors to users
#    - Log errors server-side
#    - Show user-friendly error messages


# Advanced Customizations:
# ───────────────────────

# 1. CUSTOM THEMES:
#    Create .streamlit/config.toml:
#    ```toml
#    [theme]
#    primaryColor = "#1f77b4"
#    backgroundColor = "#0e1117"
#    secondaryBackgroundColor = "#262730"
#    textColor = "#fafafa"
#    font = "sans serif"
#    ```

# 2. MULTI-PAGE APP:
#    Create pages/ directory:
#    ```
#    app.py           # Main page
#    pages/
#      1_📊_Analysis.py
#      2_📈_Charts.py
#      3_⚙️_Settings.py
#    ```
#    Streamlit automatically creates navigation

# 3. CUSTOM COMPONENTS:
#    Use streamlit-extras for enhanced UI:
#    ```python
#    from streamlit_extras.colored_header import colored_header
#    colored_header(
#        label="FinVault Analysis",
#        description="AI-powered financial insights",
#        color_name="blue-70"
#    )
#    ```

# 4. ANALYTICS TRACKING:
#    ```python
#    import time
   
#    start_time = time.time()
#    result = workflow.invoke(...)
#    duration = time.time() - start_time
   
#    # Log to analytics service
#    log_query(
#        query=prompt,
#        duration=duration,
#        user_id=st.session_state.thread_id
#    )
#    ```


# Troubleshooting Common Issues:
# ─────────────────────────────

# 1. "Script rerun requested outside of app context"
#    - Cause: Using st.rerun() outside Streamlit app
#    - Fix: Ensure all st.* calls are inside app.py

# 2. Cache not working:
#    - Cause: Function parameters are mutable objects
#    - Fix: Use @st.cache_data with hash_funcs

# 3. Session state not persisting:
#    - Cause: Using regular variables instead of session_state
#    - Fix: Always use st.session_state for persistence

# 4. Streaming not showing:
#    - Cause: No placeholder.update() in stream loop
#    - Fix: Update container on each chunk

# 5. Email validation failing:
#    - Cause: Edge cases in validation logic
#    - Fix: Use email validation library or regex


# This Streamlit app provides a production-ready foundation for deploying
# your multi-agent financial analysis system with a user-friendly interface!