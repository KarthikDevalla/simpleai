import os
import tempfile
import asyncio
from dotenv import load_dotenv
from groq import Groq
import chainlit as cl
import numpy as np
import io
import wave
import audioop

load_dotenv()
client = Groq(api_key=os.environ['GROQ_API_KEY'])

SILENCE_THRESHOLD = 5000 
MINIMUM_AUDIO_DURATION = 1.0  
MAXIMUM_AUDIO_DURATION = 30.0   



class InterviewSim:
    def __init__(self):
        self.system_prompt_cached = None
        self.job_description = None
        self.resume_text = None

    def cache_context(self, job_description: str, resume_text: str):
        """Cache the JD, resume, and create system prompt once"""
        self.job_description = job_description
        self.resume_text = resume_text
        
        # Create and cache the system prompt
        self.system_prompt_cached = {
            'role': 'system',
            'content': f"""
                    You are simulating a confident and professional candidate interviewing for a Senior Data Analyst role.
                    Respond naturally, with concise and relevant answers. Focus on demonstrating your expertise in Python, SQL, and Excel, and tailor your responses based on the job requirements.
                    Only respond after the interviewer has completed a question. Do not interrupt or react to short pauses.

                    ### Job Description:
                    {job_description}

                    ### Resume:
                    {resume_text}

                    Answer questions as if you're the candidate described in the resume and aiming for the role described in the JD. Mention specific experience, metrics, tools, or achievements when appropriate. Avoid generic answers.
                                """
        }
    
    def get_cached_system_prompt(self):
        """Return the cached system prompt"""
        return self.system_prompt_cached
    
    def is_context_cached(self):
        """Check if context is already cached"""
        return self.system_prompt_cached is not None

# Global bot instance
interview_bot = InterviewSim()

@cl.step(type="tool")
async def speech_to_text(audio_data):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data)
            tmp_file_path = tmp_file.name
        
        with open(tmp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",  
                response_format="text"     
            )
        
        os.unlink(tmp_file_path)  
        return transcription
        
    except Exception as e:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",  
                    response_format="text"     
                )
            
            os.unlink(tmp_file_path)  
            return transcription
        except:
            raise Exception(f"Speech conversion failed")

async def process_answer(transcription):
    
    message_history = cl.user_session.get("message_history", [])

    if not message_history and interview_bot.is_context_cached():
        message_history.append(interview_bot.get_cached_system_prompt())


    message_history.append({'role': 'user', 'content': transcription})

    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=message_history,
            temperature=0.2,
            max_tokens=1024,
            top_p=1,
            stop=None)    
        response = completion.choices[0].message.content
        message_history.append({'role': 'assistant', 'content': response})
        cl.user_session.set("message_history", message_history)
        return response
        
    except Exception as e:
        try:
            model_name = "llama-3.3-70b-versatile"
            completion = await client.chat.completions.create(
                model=model_name,
                messages=message_history,
                temperature=0.2,
                max_tokens=1024,
                top_p=1,
            )
        except Exception as e2:
            raise Exception(f"❌ Both models failed: {e2}")
    
    message_history.append({'role': 'assistant', 'content': response})
    cl.user_session.set("message_history", message_history)
    
    return response
        

@cl.on_message
async def on_message(message: cl.Message):
    user_input = message.content.strip()

    if not user_input:
        await cl.Message(content="Please provide a valid question.").send()
        return
    try:
        response = await process_answer(user_input)
        await cl.Message(content=response).send()

    except Exception as e:
        await cl.Message(content="Unknown Error").send()

@cl.on_chat_start
async def start():
    job_description = """This position will be part of the Commercial Analytics Hub. Leverage data analytics and data mining skills to deliver insights for the commercial organization that drive revenue increase, profitability improvement and margin expansion globally.
        - Creating analytical models
        - Understand the business objectives, do the research, structure the model and deliver outputs which focus on areas like
        - Price / Product portfolio optimization
        - Identification of profit increase opportunities: price change, Cost reduction, product mix change etc.
        - Price / PPV / Pipeline forecasting
        - Price leakage, elasticity, variance analysis
        - Automation and optimization
        - Automate the analytical models leveraging R/Python/SQL or other tools such as Power Automate / KNIME.
        - Optimize the analytics to improve efficiency
        - Performance monitoring and reporting
        - Create automated dashboards for functional stakeholders, regional / global leaders and CXOs.
        - Perform analysis for regular reporting of organizational performance (pricing, product, procurement etc.) and automate the reporting analytics
        - Mndatory skills include Python, SQL, Excel"""
    
    resume_text = """WORK EXPERIENCE 
        Source Consulting LLC, United States: Machine Learning Engineer                   
        March 2024-Present 
        • Developed a recommendation system for a grocery chain using FP-Growth algorithm, analysing customer purchasing behavior to 
        suggest products and improve user engagement 
        • Extracted and processed transactional data from AWS cloud, using AWS S3 and AWS Glue to create a robust data pipeline 
        • Coordinated within a multidisciplinary team and closely with frontend developers and data analysts to integrate the model into the 
        client’s web application and to ensure alignment with business goals 
        • Monitored and optimized model performance, achieving improvements in KPIs such as conversion rates (+10%) and average order 
        value (+15%) through periodic evaluation and fine-tuning 
        Avolta Inc, Canada: Machine Learning Engineer Intern                   
        October 2023-December 2023 
        • Engineered SOTA automotive user authentication system using One shot Computer Vision algorithms, boosting security to 90% 
        • Reduced authentication time by 64% by leveraging parallel processing, hyperparameter tuning, and A/B testing of models, 
        significantly improving user experience 
        • Implemented a distributed edge computing solution using scalable models on microcontrollers to ensure efficient load balancing 
        • Collaborated with 4 teams to develop technical documentation for the system, addressing all technical aspects including setup, 
        installation, and usage and ensuring clear instructions to support seamless implementation and user understanding 
        Cognizant Technology Solutions, India: Data Science Intern         
        December 2021-June 2022 
        • Developed an unsupervised ML model for an e-commerce client to identify at- risk customers, reducing customer churn by 8%  
        • Orchestrated an ETL pipeline using Python and SQLite to efficiently process a dataset close to 1M rows and performed statistical 
        analysis, including the Dickey-Fuller test and Kruskal Wallis test, to identify and remove seasonality from the data 
        • Automated model deployment via CI/CD pipeline (GitHub Actions) for rapid updates and optimal churn detection with new data 
        • Communicated project results and methodologies to stakeholders using interactive Tableau dashboards and Jupyter Notebooks 
        SKILLS 
        • Programming  Languages: Python, C, R, SQL 
        • Machine Learning: TensorFlow, PyTorch, Scikit Learn, Regression, Classification, Clustering, Convolutional Neural Networks 
        (CNN), Time series Forecasting, Recommender Systems, Computer Vision, Deep Learning, Natural Language Processing (NLP), 
        Reinforcement Learning, Anomaly Detection, ML Ops, LangChain, LLMs, Transformers 
        • Database: DBMS, ETL, Oracle SQL Live, MySQL, Elastic Search 
        • Data Visualization: Pandas, Matplotlib, Seaborn, Tableau, Excel, Plotly, Power BI 
        • Frameworks and Cloud: FastAPI, Flask, Streamlit, Chainlit, Hugging Face, AWS, Docker, Git, MLflow 
        PROJECTS  
        KMEngine (Python, PyPi, Docker Image, Object Oriented Programming (OOP))               
        February 2023 
        • Designed a versatile tool for streamlining core machine learning workflows which can handle both regression and classification tasks 
        with automated Extrapolatory Data Analysis (EDA) and distributed computing using Ray 
        • Constructed automated analysis pipelines at scale which cover data mining, cleaning, normalization, and  model training 
        • Empowered users to customize model parameters for a variety of machine learning tasks and facilitated model serialization and saving 
        Real Time Human Face Emotion Recognition (Python, TensorFlow, Keras, Computer Vision, Deep Learning)                 
        June 2023 
        • Built a highly efficient real-time emotion recognition system achieving a processing speed of 30 fps utilizing Haar cascades  
        • Capitalized transfer learning with a pre-trained Deep Learning model: ResNet15V2, achieving an accuracy of 94% after fine tuning 
        its performance using hyperparameter tuning and optimizing model architecture 
        • Curated and preprocessed a diverse facial emotion dataset of 8,000 annotated images using data engineering techniques 
        Gamma: A Financial Chatbot (Python, Flask, DialogFlow)        
        July 2023  
        • Created a sophisticated financial chatbot leveraging Dialogflow, empowering users to access real-time stock prices, currency exchange 
        rates, and company-specific financial news through natural language queries 
        • Employed Flask to manage over 1000 user requests, integrated external APIs for data retrieval, and iteratively refined the system 
        based on customer behaviors 
        • Extended the financial chatbot's accessibility and user reach by deploying it on the Telegram messaging platform 
        Maximum-218M: Language Model (HuggingFace, PyTorch, NLP, Large Language Models, CUDA)                        
        December 2024 
        • Coded a Large Language Model (LLM) with 218M parameters entirely from scratch to explore lightweight language models for text 
        generation. Pretrained the model on 3M tokens, ensuring a foundational understanding of language patterns 
        • Incorporated RoPE(Rotary Positional embeddings) and GeGLU activation function to improve the model's representation of positional 
        context and enhance its overall efficiency and performance 
        • Instruction fine-tuned the model to generate coherent outputs and perform various text-generation tasks effectively, achieving a Cross
        Entropy Loss of 2.07 and a Perplexity of 14, demonstrating its capability to produce high-quality results 
        EDUCATION  
        Master’s Degree: Data Science, University at Buffalo, The State University of New York  | CGPA: 3.7/4 
        January 2024 """      
    

    interview_bot.cache_context(job_description, resume_text)
    cl.user_session.set("message_history", [])
    cl.user_session.set("audio_chunks", [])
    cl.user_session.set("recording_start_time", None)
    cl.user_session.set("is_recording", False)

    await cl.Message(content="Simple AI. Fire away your questions...").send()

@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    cl.user_session.set("recording_start_time", asyncio.get_event_loop().time())
    cl.user_session.set("is_recording", True)
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):

    if not cl.user_session.get("is_recording", False):
        return
    
    audio_chunks = cl.user_session.get("audio_chunks", [])
    audio_chunk = np.frombuffer(chunk.data, dtype=np.int16)
    audio_chunks.append(audio_chunk)
    cl.user_session.set("audio_chunks", audio_chunks)
    
    start_time = cl.user_session.get("recording_start_time", 0)
    current_time = asyncio.get_event_loop().time()
    
    if current_time - start_time > MAXIMUM_AUDIO_DURATION:
        await process_audio_on_demand()

@cl.on_audio_end
async def on_audio_end():
    """Process audio when user manually stops recording"""
    cl.user_session.set("is_recording", False)
    await process_audio_on_demand()


async def process_audio_on_demand():
    audio_chunks = cl.user_session.get("audio_chunks", [])
    if not audio_chunks:
        await cl.Message(content="No audio detected. Please try again.").send()
        return
    
    try:
        # Concatenate audio chunks
        concatenated = np.concatenate(audio_chunks)
        
        # Check minimum duration
        duration = len(concatenated) / 24000.0  # Sample rate is 24000
        if duration < MINIMUM_AUDIO_DURATION:
            await cl.Message(content=f"Audio too short ({duration:.1f}s). Please speak for at least {MINIMUM_AUDIO_DURATION}s.").send()
            return

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)    # Mono
            wav_file.setsampwidth(2)    # 16-bit
            wav_file.setframerate(24000) # Sample rate
            wav_file.writeframes(concatenated.tobytes())

        wav_buffer.seek(0)
        audio_buffer = wav_buffer.getvalue()

        cl.user_session.set("audio_chunks", [])
            
        await cl.Message(content="🎧 Processing your audio...").send()
        transcription = await speech_to_text(audio_buffer)
        
        if not transcription.strip():
            await cl.Message(content="Could not transcribe audio. Please try again.").send()
            return
        
        # Show transcription
        await cl.Message(
            author="You",
            type="user_message",
            content=f"🎤 **Transcribed:** {transcription}",
        ).send()
        
        # Process and respond
        answer = await process_answer(transcription)
        await cl.Message(
            author="AI Candidate", 
            type="ai_message", 
            content=answer
        ).send()
        
    except Exception as e:
        await cl.Message(content=f"Error processing audio: {str(e)}").send()
    finally:
        # Reset recording state
        cl.user_session.set("audio_chunks", [])
        cl.user_session.set("is_recording", False)

@cl.action_callback("clear_conversation")
async def clear_conversation():
    """Clear conversation history while keeping cached context"""
    cl.user_session.set("message_history", [])
    await cl.Message(content="🔄 Conversation cleared! Context remains cached.").send()

@cl.action_callback("show_context")
async def show_context():
    """Show cached context information"""
    if interview_bot.is_context_cached():
        await cl.Message(
            content=f"📋 **Cached Context:**\n\n**Job Description:** {len(interview_bot.job_description)} characters\n**Resume:** {len(interview_bot.resume_text)} characters\n**System Prompt:** Cached ✅"
        ).send()
    else:
        await cl.Message(content="❌ No context cached").send()