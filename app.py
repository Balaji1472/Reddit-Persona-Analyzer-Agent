"""
Reddit Persona Analyzer - A Streamlit application for analyzing Reddit users and 
subreddits with detailed persona generation and citation tracking.

This module provides functionality to:
1. Scrape Reddit user posts and comments
2. Generate detailed user personas with citations
3. Analyze subreddit communities
4. Export results with full citation references
"""

import json
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import praw
import google.generativeai as genai

import streamlit as st
from dotenv import load_dotenv

# Load environment variables - Updated for Streamlit Cloud
try:
    load_dotenv()
except:
    pass  # In Streamlit Cloud, we'll use st.secrets

def get_reddit_credentials():
    try:
        # Try Streamlit secrets first (for cloud deployment)
        return {
            'client_id': st.secrets.get("REDDIT_CLIENT_ID", os.getenv("REDDIT_CLIENT_ID")),
            'client_secret': st.secrets.get("REDDIT_CLIENT_SECRET", os.getenv("REDDIT_CLIENT_SECRET")),
            'user_agent': st.secrets.get("REDDIT_USER_AGENT", os.getenv("REDDIT_USER_AGENT"))
        }
    except:
        # Fallback to environment variables
        return {
            'client_id': os.getenv("REDDIT_CLIENT_ID"),
            'client_secret': os.getenv("REDDIT_CLIENT_SECRET"),
            'user_agent': os.getenv("REDDIT_USER_AGENT")
        }

# Initialize session state for storing data
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {}

# Configuration constants
MAX_CONTENT_LENGTH = 500
DEFAULT_MAX_POSTS = 10
DEFAULT_MAX_COMMENTS = 15
MAX_ANALYSIS_ITEMS = 30
INITIAL_RETRY_DELAY = 1
MAX_RETRIES = 3

# Initialize Reddit API client - Updated
try:
    creds = get_reddit_credentials()
    reddit = praw.Reddit(
        client_id=creds['client_id'],
        client_secret=creds['client_secret'],
        user_agent=creds['user_agent']
    )
except Exception as e:
    st.error(f"Failed to initialize Reddit client: {str(e)}")
    reddit = None


class RedditPersonaAnalyzer:
    """
    A class for analyzing Reddit users and subreddits to generate detailed personas
    and community insights with citation tracking.
    """
    
    def __init__(self):
        """Initialize the analyzer with Gemini AI model."""
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Failed to initialize Gemini model: {str(e)}")
            self.model = None
        
    def extract_identifier(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract username or subreddit name from Reddit URL.
        
        Args:
            url: Reddit URL to parse
            
        Returns:
            Tuple of (type, identifier) where type is 'user' or 'subreddit'
        """
        if not url:
            return None, None
            
        user_pattern = r'reddit\.com/u(?:ser)?/([^/]+)'
        subreddit_pattern = r'reddit\.com/r/([^/]+)'
        
        user_match = re.search(user_pattern, url)
        if user_match:
            return 'user', user_match.group(1)
            
        subreddit_match = re.search(subreddit_pattern, url)
        if subreddit_match:
            return 'subreddit', subreddit_match.group(1)
            
        return None, None
    
    def clean_content(self, text: Any) -> str:
        """
        Clean and sanitize text content from Reddit posts/comments.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not text:
            return ""
        
        # handle cases where text might be None or not a string
        text = str(text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text[:MAX_CONTENT_LENGTH] if len(text) > MAX_CONTENT_LENGTH else text
    
    def retry_with_backoff(self, func, max_retries: int = MAX_RETRIES, 
                          initial_delay: float = INITIAL_RETRY_DELAY):
        """
        Retry function with exponential backoff for handling API rate limits.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                delay = initial_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    
    def scrape_user_data(self, username: str, max_posts: int = DEFAULT_MAX_POSTS, 
                        max_comments: int = DEFAULT_MAX_COMMENTS) -> Tuple[List[Dict], List[Dict]]:
        """
        Scrape posts and comments from a Reddit user.
        
        Args:
            username: Reddit username to scrape
            max_posts: Maximum number of posts to retrieve
            max_comments: Maximum number of comments to retrieve
            
        Returns:
            Tuple of (posts_list, comments_list) containing scraped data
        """
        if not reddit:
            st.error("Reddit client not initialized")
            return [], []
            
        def _scrape():
            try:
                user = reddit.redditor(username)
                
                posts = []
                for idx, post in enumerate(user.submissions.new(limit=max_posts)):
                    posts.append({
                        'type': 'post',
                        'title': self.clean_content(post.title),
                        'content': self.clean_content(post.selftext),
                        'subreddit': post.subreddit.display_name,
                        'score': post.score,
                        'created_utc': post.created_utc,
                        'url': f"https://reddit.com{post.permalink}",
                        'id': f"post_{idx + 1}",
                        'full_content': (
                            f"Title: {self.clean_content(post.title)}\n"
                            f"Content: {self.clean_content(post.selftext)}"
                        )
                    })
                
                comments = []
                for idx, comment in enumerate(user.comments.new(limit=max_comments)):
                    comments.append({
                        'type': 'comment',
                        'content': self.clean_content(comment.body),
                        'subreddit': comment.subreddit.display_name,
                        'score': comment.score,
                        'created_utc': comment.created_utc,
                        'url': f"https://reddit.com{comment.permalink}",
                        'id': f"comment_{idx + 1}",
                        'full_content': self.clean_content(comment.body)
                    })
                
                return posts, comments
                
            except Exception as e:
                st.error(f"Error accessing user data: {str(e)}")
                return [], []
        
        try:
            return self.retry_with_backoff(_scrape)
        except Exception as e:
            st.error(f"Error scraping user data: {str(e)}")
            return [], []
    
    def get_top_contributors(self, subreddit_name: str, 
                           limit: int = 100) -> List[Tuple[str, float, Dict]]:
        """
        Get top contributors for a subreddit based on activity metrics.
        
        Args:
            subreddit_name: Name of the subreddit
            limit: Maximum number of posts to analyze
            
        Returns:
            List of tuples (username, score, metrics) sorted by activity
        """
        if not reddit:
            st.error("Reddit client not initialized")
            return []
            
        try:
            def _get_contributors():
                subreddit = reddit.subreddit(subreddit_name)
                
                user_metrics = defaultdict(
                    lambda: {
                        'posts': 0,
                        'comments': 0,
                        'post_score': 0,
                        'comment_score': 0
                    }
                )
                
                # Get posts
                for post in subreddit.hot(limit=limit):
                    if post.author:
                        user_metrics[post.author.name]['posts'] += 1
                        user_metrics[post.author.name]['post_score'] += post.score
                
                # Get comments
                for post in subreddit.hot(limit=limit//2):
                    try:
                        post.comments.replace_more(limit=0)
                        for comment in post.comments.list()[:50]:
                            if comment.author:
                                username = comment.author.name
                                user_metrics[username]['comments'] += 1
                                user_metrics[username]['comment_score'] += comment.score
                    except Exception:
                        continue  # Skip problematic posts
                
                scored_users = []
                for username, metrics in user_metrics.items():
                    combined_score = (
                        metrics['posts'] * 3 +
                        metrics['comments'] * 1 +
                        metrics['post_score'] * 0.1 +
                        metrics['comment_score'] * 0.05
                    )
                    scored_users.append((username, combined_score, metrics))
                
                return sorted(scored_users, key=lambda x: x[1], reverse=True)[:5]
            
            return self.retry_with_backoff(_get_contributors)
            
        except Exception as e:
            st.error(f"Error getting top contributors: {str(e)}")
            return []
    
    def prepare_content_for_analysis(self, posts: List[Dict], 
                                   comments: List[Dict], 
                                   max_items: int = MAX_ANALYSIS_ITEMS) -> List[Dict]:
        """
        Prepare content for AI analysis with proper structure.
        
        Args:
            posts: List of post dictionaries
            comments: List of comment dictionaries
            max_items: Maximum number of items to include
            
        Returns:
            List of content items for analysis
        """
        all_content = []
        
        for post in posts[:max_items//2]:
            all_content.append({
                'id': post['id'],
                'type': 'post',
                'content': post['full_content'],
                'subreddit': post['subreddit'],
                'url': post['url'],
                'score': post['score']
            })
        
        for comment in comments[:max_items//2]:
            all_content.append({
                'id': comment['id'],
                'type': 'comment',
                'content': comment['full_content'],
                'subreddit': comment['subreddit'],
                'url': comment['url'],
                'score': comment['score']
            })
        
        return all_content
    
    def generate_persona_with_citations(self, posts: List[Dict], 
                                      comments: List[Dict], 
                                      username: str) -> Optional[str]:
        """
        Generate detailed persona with citation tracking.
        
        Args:
            posts: List of user posts
            comments: List of user comments
            username: Reddit username
            
        Returns:
            Generated persona text with citations or None if failed
        """
        if not self.model:
            st.error("Gemini model not initialized")
            return None
            
        content_for_analysis = self.prepare_content_for_analysis(posts, comments)
        
        if not content_for_analysis:
            st.warning("No content available for analysis")
            return None
        
        prompt = f"""
        Generate a detailed Reddit user persona for '{username}' using the content below. 
        For EACH characteristic you identify, you MUST cite the specific content ID(s) 
        that support that characteristic.

        Content for Analysis:
        {json.dumps(content_for_analysis, indent=2)}

        Create the persona in the following format with citations:

        ## User Persona: {username}

        ### Profile Overview
        - **Age Range:** [Your assessment] 
          Citations: [List content IDs that support this, e.g., post_1, comment_3]
        - **Location:** [Your assessment]
          Citations: [List content IDs that support this]
        - **Occupation:** [Your assessment]
          Citations: [List content IDs that support this]

        ### Interests & Hobbies
        - **Primary Interests:** [List main interests]
          Citations: [List content IDs for each interest]
        - **Most Active Subreddits:** [List top subreddits]
          Citations: [List content IDs showing activity in these subreddits]

        ### Personality Traits
        - **Communication Style:** [Your assessment]
          Citations: [List content IDs that demonstrate this style]
        - **Emotional Tone:** [Your assessment]
          Citations: [List content IDs showing this tone]
        - **Social Engagement Level:** [Your assessment]
          Citations: [List content IDs showing engagement patterns]

        ### Behavioral Patterns
        - **Posting Frequency:** [Your assessment]
          Citations: [List content IDs that indicate posting habits]
        - **Content Preferences:** [Your assessment]
          Citations: [List content IDs showing preferred content types]
        - **Interaction Style:** [Your assessment]
          Citations: [List content IDs showing how they interact]

        ### Goals & Motivations
        - **Primary Drivers:** [Your assessment]
          Citations: [List content IDs that reveal motivations]
        - **Community Involvement:** [Your assessment]
          Citations: [List content IDs showing community participation]

        ### Pain Points & Challenges
        - **Common Concerns:** [Your assessment]
          Citations: [List content IDs discussing these concerns]
        - **Challenges Discussed:** [Your assessment]
          Citations: [List content IDs mentioning challenges]

        IMPORTANT INSTRUCTIONS:
        1. Every characteristic MUST have at least one citation
        2. Use the exact content IDs from the provided data (e.g., post_1, comment_5)
        3. If you cannot find evidence for a characteristic, state "No evidence found"
        4. Be specific about which content supports each claim
        5. Limit the output to 500-600 words total
        6. Make citations format consistent: Citations: [post_1, comment_3, post_5]
        """
        
        try:
            def _generate():
                response = self.model.generate_content(prompt)
                return response.text
            
            return self.retry_with_backoff(_generate)
            
        except Exception as e:
            st.error(f"Error generating persona: {str(e)}")
            return None
    
    def generate_subreddit_analysis(self, subreddit_name: str, 
                                  top_contributors: List[Tuple]) -> Optional[str]:
        """
        Generate comprehensive subreddit community analysis.
        
        Args:
            subreddit_name: Name of the subreddit
            top_contributors: List of top contributor data
            
        Returns:
            Generated analysis text or None if failed
        """
        if not self.model:
            st.error("Gemini model not initialized")
            return None
            
        if not top_contributors:
            st.warning("No contributors data available for analysis")
            return None
        
        prompt = f"""
        Analyze r/{subreddit_name} based on its top 5 contributors and create a 
        community analysis.

        Top Contributors:
        {json.dumps([{'username': user[0], 'score': user[1], 'metrics': user[2]} 
                    for user in top_contributors], indent=2)}

        Create a comprehensive analysis:
        ## Subreddit Analysis: r/{subreddit_name}

        ### Community Overview
        - Main topics and themes
        - Community size and activity level

        ### Top Contributors Profile
        - Brief description of each top contributor
        - Their role in the community

        ### Community Dynamics
        - Interaction patterns
        - Content preferences
        - Engagement levels

        ### Community Culture
        - Communication style
        - Values and norms
        - Common interests

        ### Insights & Recommendations
        - Key findings about the community
        - Suggestions for engagement

        Keep response under 400-500 words and avoid long paragraphs. 
        Keep each section clear and detailed.
        """
        
        try:
            def _generate():
                response = self.model.generate_content(prompt)
                return response.text
            
            return self.retry_with_backoff(_generate)
            
        except Exception as e:
            st.error(f"Error generating subreddit analysis: {str(e)}")
            return None
    
    def create_citation_reference(self, posts: List[Dict], 
                                comments: List[Dict]) -> str:
        """
        Create a reference guide for citations.
        
        Args:
            posts: List of post dictionaries
            comments: List of comment dictionaries
            
        Returns:
            Formatted citation reference text
        """
        reference_text = "\n\n" + "="*50 + "\n"
        reference_text += "CITATION REFERENCE\n"
        reference_text += "="*50 + "\n\n"
        
        if posts:
            reference_text += "POSTS:\n"
            reference_text += "-"*30 + "\n"
            for post in posts:
                reference_text += f"[{post['id']}] r/{post['subreddit']} - {post['title']}\n"
                reference_text += f"URL: {post['url']}\n"
                content_preview = post['content'][:100]
                if len(post['content']) > 100:
                    content_preview += "..."
                reference_text += f"Score: {post['score']} | Content: {content_preview}\n\n"
        
        if comments:
            reference_text += "\nCOMMENTS:\n"
            reference_text += "-"*30 + "\n"
            for comment in comments:
                reference_text += f"[{comment['id']}] r/{comment['subreddit']}\n"
                reference_text += f"URL: {comment['url']}\n"
                content_preview = comment['content'][:100]
                if len(comment['content']) > 100:
                    content_preview += "..."
                reference_text += f"Score: {comment['score']} | Content: {content_preview}\n\n"
        
        return reference_text
    
    def save_to_temp_file(self, content: str, identifier: str, 
                         analysis_type: str, posts: Optional[List[Dict]] = None, 
                         comments: Optional[List[Dict]] = None) -> Tuple[str, str]:
        """
        Save analysis content to temporary file.
        
        Args:
            content: Content to save
            identifier: User/subreddit identifier
            analysis_type: Type of analysis (persona/subreddit)
            posts: Optional list of posts for citation reference
            comments: Optional list of comments for citation reference
            
        Returns:
            Tuple of (temp_path, filename)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{identifier}_{analysis_type}_{timestamp}.txt"
        
        # Add citation reference for user analysis
        if analysis_type == "persona" and posts is not None and comments is not None:
            citation_reference = self.create_citation_reference(posts, comments)
            content = content + citation_reference
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', 
                                           delete=False, encoding='utf-8') as f:
                f.write(content)
                temp_path = f.name
            
            return temp_path, filename
            
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return "", filename


def initialize_gemini_api(api_key: str) -> bool:
    """
    Initialize Gemini API with provided key.
    
    Args:
        api_key: Gemini API key
        
    Returns:
        True if initialization successful, False otherwise
    """
    try:
        genai.configure(api_key=api_key)
        # Test the API key with a simple request
        test_model = genai.GenerativeModel('gemini-1.5-flash')
        _ = test_model.generate_content("Hi").text
        return True
    except Exception as e:
        st.error(f"API initialization failed: {str(e)}")
        return False


def display_user_results():
    """Display user analysis results from session state."""
    data = st.session_state.analysis_data
    
    st.header("📊 User Analysis Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Generated Persona with Citations")
        st.markdown(data['persona_text'])
        
        # Show citation note
        st.info("💡 **Citation System:** Each characteristic includes citations "
                "(e.g., post_1, comment_3) that reference specific content. "
                "The full citation reference is included in the downloaded file.")
    
    with col2:
        st.subheader("Data Summary")
        st.metric("Posts Analyzed", len(data['posts']))
        st.metric("Comments Analyzed", len(data['comments']))
        
        try:
            with open(data['temp_path'], 'r', encoding='utf-8') as f:
                st.download_button(
                    label="📥 Download Persona with Citations",
                    data=f.read(),
                    file_name=data['filename'],
                    mime="text/plain",
                    help="Download includes the persona analysis and complete citation reference"
                )
        except Exception as e:
            st.error(f"Error reading file for download: {str(e)}")


def display_sample_data():
    """Display sample data if requested."""
    data = st.session_state.analysis_data
    posts = data['posts']
    comments = data['comments']
    
    st.header("📄 Sample Data")
    
    if posts:
        st.subheader("Sample Posts")
        for post in posts[:3]:
            post_title = post['title']
            if len(post_title) > 50:
                post_title = post_title[:50] + "..."
            with st.expander(f"[{post['id']}] {post_title}"):
                st.write(f"**Subreddit:** r/{post['subreddit']}")
                st.write(f"**Score:** {post['score']}")
                st.write(f"**Content:** {post['content']}")
                st.write(f"**URL:** {post['url']}")
    else:
        st.write("No posts found for this user.")
    
    if comments:
        st.subheader("Sample Comments")
        for comment in comments[:3]:
            with st.expander(f"[{comment['id']}] Comment in r/{comment['subreddit']}"):
                st.write(f"**Score:** {comment['score']}")
                st.write(f"**Content:** {comment['content']}")
                st.write(f"**URL:** {comment['url']}")
    else:
        st.write("No comments found for this user.")


def display_subreddit_results():
    """Display subreddit analysis results from session state."""
    data = st.session_state.analysis_data
    
    st.header("📊 Subreddit Analysis Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Community Analysis")
        st.markdown(data['analysis_text'])
    
    with col2:
        st.subheader("Top Contributors")
        for i, (username, score, metrics) in enumerate(data['top_contributors'], 1):
            with st.expander(f"{i}. u/{username}"):
                st.write(f"**Combined Score:** {score:.1f}")
                st.write(f"**Posts:** {metrics['posts']}")
                st.write(f"**Comments:** {metrics['comments']}")
                st.write(f"**Post Score:** {metrics['post_score']}")
                st.write(f"**Comment Score:** {metrics['comment_score']}")
        
        try:
            with open(data['temp_path'], 'r', encoding='utf-8') as f:
                st.download_button(
                    label="📥 Download Analysis",
                    data=f.read(),
                    file_name=data['filename'],
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"Error reading file for download: {str(e)}")


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Reddit Persona Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Reddit Persona Analyzer")
    st.markdown("Analyze Reddit users or subreddits and generate detailed insights with citations.")
    
    # API Key Configuration
    st.sidebar.header("🔐 Gemini API Key")
    user_api_key = st.sidebar.text_input(
        "Enter your Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key here",
    )

    if user_api_key:
        if initialize_gemini_api(user_api_key):
            st.session_state['gemini_key'] = user_api_key
            st.sidebar.success("✅ API Key is valid")
        else:
            st.sidebar.error("❌ Invalid key or quota error")
            st.stop()
    else:
        st.sidebar.warning("⚠️ Please enter your Gemini API key to continue.")
        st.stop()

    # Check Reddit API initialization
    if not reddit:
        st.error("❌ Reddit API not properly configured. Please check your secrets configuration.")
        st.info("Make sure you have set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT in your Streamlit secrets.")
        st.stop()

    # Initialize analyzer
    analyzer = RedditPersonaAnalyzer()
    
    # Configuration
    st.sidebar.header("Configuration")
    analysis_mode = st.sidebar.radio("Analysis Mode", ["Analyze User", "Analyze Subreddit"])
    
    if analysis_mode == "Analyze User":
        max_posts = st.sidebar.slider("Max Posts", 5, 25, DEFAULT_MAX_POSTS)
        max_comments = st.sidebar.slider("Max Comments", 10, 30, DEFAULT_MAX_COMMENTS)
        
        profile_url = st.text_input(
            "Reddit Profile URL:",
            placeholder="https://www.reddit.com/user/username",
            help="Enter Reddit profile URL"
        )
    else:
        profile_url = st.text_input(
            "Subreddit URL:",
            placeholder="https://www.reddit.com/r/subreddit",
            help="Enter subreddit URL"
        )
    
    # Show Sample Data checkbox for user analysis
    show_sample_data = False
    if (st.session_state.analysis_data.get('analysis_complete', False) and 
        st.session_state.analysis_data['type'] == 'user'):
        st.sidebar.header("View Options")
        show_sample_data = st.sidebar.checkbox("Show Sample Data")
    
    # Analysis button
    analyze_button = st.button("🔍 Start Analysis", type="primary")
    
    if analyze_button and profile_url:
        url_type, identifier = analyzer.extract_identifier(profile_url)
        
        if not url_type:
            st.error("Invalid URL format. Please check and try again.")
            return
        
        if ((analysis_mode == "Analyze User" and url_type != "user") or 
            (analysis_mode == "Analyze Subreddit" and url_type != "subreddit")):
            st.error(f"URL type mismatch. Expected {analysis_mode.lower()}, got {url_type}.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if analysis_mode == "Analyze User":
            st.success(f"Analyzing user: u/{identifier}")
            
            status_text.text("🔄 Scraping user data...")
            progress_bar.progress(25)
            
            posts, comments = analyzer.scrape_user_data(identifier, max_posts, max_comments)
            
            if not posts and not comments:
                st.error("No data found. User might be private or inactive.")
                return
            
            status_text.text("🤖 Generating persona with citations...")
            progress_bar.progress(75)
            
            persona_text = analyzer.generate_persona_with_citations(posts, comments, identifier)
            
            if not persona_text:
                st.error("Failed to generate persona. Please try again.")
                return
            
            status_text.text("💾 Saving results...")
            progress_bar.progress(100)
            
            temp_path, filename = analyzer.save_to_temp_file(
                persona_text, identifier, "persona", posts, comments
            )
            
            # Store data in session state
            st.session_state.analysis_data = {
                'type': 'user',
                'identifier': identifier,
                'posts': posts,
                'comments': comments,
                'persona_text': persona_text,
                'temp_path': temp_path,
                'filename': filename,
                'analysis_complete': True
            }
            
            status_text.text("✅ Analysis complete!")
        
        else:  # Subreddit analysis
            st.success(f"Analyzing subreddit: r/{identifier}")
            
            status_text.text("🔄 Finding top contributors...")
            progress_bar.progress(30)
            
            top_contributors = analyzer.get_top_contributors(identifier)
            
            if not top_contributors:
                st.error("No contributors found. Subreddit might be private or inactive.")
                return
            
            status_text.text("🤖 Generating subreddit analysis...")
            progress_bar.progress(75)
            
            analysis_text = analyzer.generate_subreddit_analysis(identifier, top_contributors)
            
            if not analysis_text:
                st.error("Failed to generate analysis. Please try again.")
                return
            
            status_text.text("💾 Saving results...")
            progress_bar.progress(100)
            
            temp_path, filename = analyzer.save_to_temp_file(
                analysis_text, identifier, "subreddit"
            )
            
            # Store data in session state
            st.session_state.analysis_data = {
                'type': 'subreddit',
                'identifier': identifier,
                'top_contributors': top_contributors,
                'analysis_text': analysis_text,
                'temp_path': temp_path,
                'filename': filename,
                'analysis_complete': True
            }
            
            status_text.text("✅ Analysis complete!")
    
    # Display results if analysis is complete
    if st.session_state.analysis_data.get('analysis_complete', False):
        if st.session_state.analysis_data['type'] == 'user':
            display_user_results()
            
            # Show sample data if requested
            if show_sample_data:
                display_sample_data()
        
        else:  # Subreddit analysis
            display_subreddit_results()


if __name__ == "__main__":
    main()
