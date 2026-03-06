"""
Streamlit Chat UI for MultiModal RAG System
"""

import streamlit as st
import requests
import json
from typing import Optional, List, Dict
from datetime import datetime

# Page config
st.set_page_config(
    page_title='MultiModal RAG Chatbot',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Styling
st.markdown('''
<style>
    .main { max-width: 1200px; }
    .chat-message { 
        padding: 1rem; 
        margin-bottom: 1rem; 
        border-radius: 0.5rem;
        background-color: #f0f2f6;
    }
    .user-message { 
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message { 
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .source-citation {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
        padding: 0.5rem;
        background-color: #fafafa;
        border-left: 3px solid #1f77b4;
    }
</style>
''', unsafe_allow_html=True)

# Session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'api_url' not in st.session_state:
    st.session_state.api_url = 'http://localhost:8000'

# Sidebar
with st.sidebar:
    st.title('⚙️ Configuration')
    
    api_url = st.text_input(
        'API URL',
        value=st.session_state.api_url,
        help='FastAPI backend URL'
    )
    st.session_state.api_url = api_url
    
    st.divider()
    
    st.subheader('Retrieval Settings')
    top_k = st.slider(
        'Top K documents',
        min_value=5,
        max_value=50,
        value=10,
        help='Number of documents to retrieve'
    )
    rerank_top_n = st.slider(
        'Rerank Top N',
        min_value=3,
        max_value=20,
        value=5,
        help='Final number of results after reranking'
    )
    
    st.divider()
    
    st.subheader('LLM Settings')
    force_provider = st.selectbox(
        'LLM Provider',
        options=['Auto', 'Ollama', 'Groq'],
        help='Which LLM to use'
    )
    force_provider_map = {'Auto': None, 'Ollama': 'ollama', 'Groq': 'groq'}
    force_provider = force_provider_map[force_provider]
    
    use_cache = st.checkbox('Use cache', value=True)
    
    st.divider()
    
    # Health check
    st.subheader('System Status')
    if st.button('🔍 Check API Health'):
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                st.success('✅ API is running')
                st.json(health)
            else:
                st.error(f'❌ API error: {response.status_code}')
        except Exception as e:
            st.error(f'❌ Connection failed: {str(e)}')
    
    st.divider()
    
    # Cache management
    st.subheader('Cache Management')
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('📊 Cache Stats'):
            try:
                response = requests.get(f"{api_url}/cache/stats", timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    st.info(f"Cached queries: {stats['size']}")
            except Exception as e:
                st.error(f'Error: {str(e)}')
    
    with col2:
        if st.button('🗑️ Clear Cache'):
            try:
                response = requests.post(f"{api_url}/cache/clear", timeout=5)
                if response.status_code == 200:
                    st.success('Cache cleared!')
            except Exception as e:
                st.error(f'Error: {str(e)}')

# Main content
st.title('🤖 MultiModal RAG Chatbot')

st.markdown('''
This chatbot retrieves information from your document collection (PDFs, images, CSV files) 
and generates answers using advanced language models with proper source citations.

**Features:**
- 🔍 Hybrid retrieval (keyword + semantic search)
- 📚 Multi-document source citations
- 🚀 Local (Ollama) and cloud (Groq) LLM support
- 💾 Response caching
- ✨ Cross-encoder reranking
''')

st.divider()

# Chat history
st.subheader('Chat History')

for msg in st.session_state.messages:
    if msg['role'] == 'user':
        with st.chat_message('user', avatar='👤'):
            st.write(msg['content'])
    else:
        with st.chat_message('assistant', avatar='🤖'):
            st.write(msg['content'])
            
            if 'sources' in msg:
                with st.expander(f"📚 Sources ({len(msg['sources'])} documents)"):
                    for i, source in enumerate(msg['sources'], 1):
                        st.markdown(f'''
**Source {i}: {source['source_file']}**
- Page: {source['page']}
- Type: {source['doc_type']}
- Preview: {source['text'][:200]}...

---
''')
            
            if 'metadata' in msg:
                meta = msg['metadata']
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"LLM: {meta.get('provider', 'unknown')}")
                with col2:
                    st.caption(f"Model: {meta.get('model', 'unknown')}")
                with col3:
                    cached = "✅ Cached" if meta.get('cached') else "🆕 Fresh"
                    st.caption(cached)

st.divider()

# Input area
st.subheader('Ask a Question')

prompt = st.chat_input(
    'Ask a question about your documents...',
    key='chat_input'
)

if prompt:
    # Add user message to history
    st.session_state.messages.append({
        'role': 'user',
        'content': prompt
    })
    
    with st.chat_message('user', avatar='👤'):
        st.write(prompt)
    
    # Process query
    with st.spinner('Retrieving documents and generating answer...'):
        try:
            response = requests.post(
                f"{st.session_state.api_url}/query",
                json={
                    'query': prompt,
                    'top_k': top_k,
                    'rerank_top_n': rerank_top_n,
                    'force_provider': force_provider,
                    'use_cache': use_cache
                },
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                
                with st.chat_message('assistant', avatar='🤖'):
                    # Display answer
                    st.write(data['answer'])
                    
                    # Display sources
                    if data['sources']:
                        with st.expander(f"📚 Sources ({len(data['sources'])} documents)"):
                            for i, source in enumerate(data['sources'], 1):
                                st.markdown(f'''
**Source {i}**
- **File:** {source['source_file']}
- **Page:** {source['page']}
- **Type:** {source['doc_type']}
- **Content:** {source['text'][:200]}...
''')
                    
                    # Display metadata
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        cached_badge = "✅ Cached" if data['cached'] else "🆕 Fresh"
                        st.metric("Cache", cached_badge)
                    with col2:
                        st.metric("Provider", data['provider'].upper())
                    with col3:
                        st.metric("Model", data['model'].split('/')[-1][:15])
                    with col4:
                        st.metric("Sources", len(data['sources']))
                
                # Add assistant message to history
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': data['answer'],
                    'sources': data['sources'],
                    'metadata': {
                        'provider': data['provider'],
                        'model': data['model'],
                        'cached': data['cached'],
                        'timestamp': datetime.now().isoformat()
                    }
                })
                
                # Rerun to update chat
                st.rerun()
            
            elif response.status_code == 500:
                st.error(f"API Error: {response.json()['detail']}")
            else:
                st.error(f"Unexpected error: {response.status_code}")
        
        except requests.exceptions.Timeout:
            st.error("Request timeout - API took too long to respond")
        except requests.exceptions.ConnectionError:
            st.error(f"Connection failed - Is API running at {st.session_state.api_url}?")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown('''
---
**MultiModal RAG Chatbot** | Powered by FAISS, BM25, Ollama, Groq, and Streamlit
''')
