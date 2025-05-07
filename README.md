# AI-Powered-Job-Application-Enhancer
AI-powered Streamlit app to optimize resumes for ATS. Analyzes resume-job fit with OpenAI GPT, offers match scores, skill gaps, keyword visuals, tailored resumes, cover letters, and interview prep. Supports PDF/DOCX/TXT, batch processing, and downloads. Ideal for job seekers.
# Resume ATS Optimizer

An advanced, AI-powered Streamlit application designed to help job seekers optimize their resumes for Applicant Tracking Systems (ATS) by analyzing and matching resumes against job descriptions. The application provides detailed match analysis, personalized recommendations, optimized resume generation, cover letter creation, and interview preparation tools.

## Features

- **Resume and Job Description Upload**: Supports PDF, DOCX, and TXT file formats for resume and job description uploads.
- **AI-Powered Analysis**: Uses OpenAI's GPT models to extract skills, calculate match scores, and provide detailed analysis of resume-job description alignment.
- **Customizable Weights**: Allows users to adjust the importance of skill match vs. content match in the analysis.
- **Keyword Density Visualization**: Displays a comparison of keyword frequency in the resume and job description using interactive Plotly charts.
- **Venn Diagram for Skills**: Visualizes common and missing skills between the resume and job description.
- **Batch Processing**: Enables comparison of a single resume against multiple job descriptions.
- **Optimized Resume Generation**: Creates a tailored resume with highlighted changes to improve ATS compatibility.
- **Cover Letter Generator**: Produces professional, job-specific cover letters.
- **Interview Preparation**: Generates potential interview questions with sample answers and a mock interview simulator.
- **Download Options**: Supports downloading outputs in Markdown, PDF, and DOCX formats.
- **Dark/Light Mode**: Toggle between themes for better user experience.
- **Caching and Rate Limiting**: Implements caching for faster performance and rate limiting for API calls to prevent overuse.
- **Resume History**: Stores previous resume versions for easy access.

## Prerequisites

- Python 3.8+
- OpenAI API key (stored in a `.env` file or provided via the sidebar)
- Required Python packages (listed in `requirements.txt`)

