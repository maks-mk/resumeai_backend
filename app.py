from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ResumeAI Backend API",
    description="AI-powered resume analysis and optimization service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models ====================

class SkillItem(BaseModel):
    name: str = Field(..., min_length=1, description="Skill name")
    proficiency: str = Field(default="intermediate", description="Proficiency level")
    years: Optional[float] = Field(default=None, description="Years of experience")

    class Config:
        example = {
            "name": "Python",
            "proficiency": "expert",
            "years": 5.0
        }


class ExperienceItem(BaseModel):
    title: str = Field(..., min_length=1, description="Job title")
    company: str = Field(..., min_length=1, description="Company name")
    duration: str = Field(..., description="Duration of employment")
    description: Optional[str] = Field(default=None, description="Job description")
    skills_used: Optional[List[str]] = Field(default=[], description="Skills used")

    class Config:
        example = {
            "title": "Senior Developer",
            "company": "Tech Company",
            "duration": "2020-2023",
            "description": "Led development of cloud applications",
            "skills_used": ["Python", "AWS", "Docker"]
        }


class EducationItem(BaseModel):
    degree: str = Field(..., min_length=1, description="Degree obtained")
    institution: str = Field(..., min_length=1, description="Educational institution")
    graduation_year: int = Field(..., description="Year of graduation")
    gpa: Optional[float] = Field(default=None, ge=0, le=4, description="GPA")

    class Config:
        example = {
            "degree": "Bachelor of Science",
            "institution": "University of Technology",
            "graduation_year": 2019,
            "gpa": 3.8
        }


class ResumeInput(BaseModel):
    name: str = Field(..., min_length=1, description="Candidate name")
    email: str = Field(..., description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    summary: Optional[str] = Field(default=None, description="Professional summary")
    skills: List[SkillItem] = Field(default=[], description="List of skills")
    experience: List[ExperienceItem] = Field(default=[], description="Work experience")
    education: List[EducationItem] = Field(default=[], description="Education details")

    class Config:
        example = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1-555-0123",
            "summary": "Experienced software engineer",
            "skills": [
                {"name": "Python", "proficiency": "expert", "years": 5}
            ],
            "experience": [
                {
                    "title": "Senior Developer",
                    "company": "Tech Corp",
                    "duration": "2020-2023",
                    "description": "Led development",
                    "skills_used": ["Python", "AWS"]
                }
            ],
            "education": [
                {
                    "degree": "B.S. Computer Science",
                    "institution": "MIT",
                    "graduation_year": 2018,
                    "gpa": 3.9
                }
            ]
        }


class ResumeAnalysis(BaseModel):
    resume_id: str = Field(..., description="Unique resume identifier")
    candidate_name: str = Field(..., description="Candidate name")
    score: float = Field(..., ge=0, le=100, description="Resume score (0-100)")
    strengths: List[str] = Field(description="Resume strengths")
    improvements: List[str] = Field(description="Areas for improvement")
    recommendations: List[str] = Field(description="Specific recommendations")

    class Config:
        example = {
            "resume_id": "res_123456",
            "candidate_name": "John Doe",
            "score": 85.5,
            "strengths": ["Strong technical skills", "Clear career progression"],
            "improvements": ["Add more quantifiable achievements", "Improve formatting"],
            "recommendations": ["Highlight project outcomes", "Add metrics to experience"]
        }


class OptimizationSuggestion(BaseModel):
    section: str = Field(..., description="Resume section")
    current_text: str = Field(..., description="Current text")
    suggested_text: str = Field(..., description="Suggested improved text")
    reason: str = Field(..., description="Reason for suggestion")

    class Config:
        example = {
            "section": "summary",
            "current_text": "Worked as a developer",
            "suggested_text": "Led development of 5+ cloud-based applications serving 100k+ users",
            "reason": "More specific and achievement-focused"
        }


# ==================== API Endpoints ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "online",
        "service": "ResumeAI Backend",
        "version": "1.0.0"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/analyze", response_model=ResumeAnalysis, tags=["Resume Analysis"])
async def analyze_resume(resume: ResumeInput) -> ResumeAnalysis:
    """
    Analyze a resume and provide a comprehensive score and feedback.
    
    - **resume**: Resume data in JSON format
    
    Returns analysis with score and recommendations.
    """
    try:
        logger.info(f"Analyzing resume for {resume.name}")
        
        # Validate resume has required sections
        if not resume.experience and not resume.education:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume must contain at least experience or education information"
            )
        
        # Calculate score based on resume completeness
        score = calculate_resume_score(resume)
        
        # Generate analysis
        strengths = identify_strengths(resume)
        improvements = identify_improvements(resume)
        recommendations = generate_recommendations(resume)
        
        analysis = ResumeAnalysis(
            resume_id=f"res_{hash(resume.email) % 1000000:06d}",
            candidate_name=resume.name,
            score=score,
            strengths=strengths,
            improvements=improvements,
            recommendations=recommendations
        )
        
        logger.info(f"Analysis complete for {resume.name} with score {score}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing resume: {str(e)}"
        )


@app.post("/optimize", response_model=List[OptimizationSuggestion], tags=["Resume Optimization"])
async def optimize_resume(resume: ResumeInput) -> List[OptimizationSuggestion]:
    """
    Get specific optimization suggestions for resume sections.
    
    - **resume**: Resume data in JSON format
    
    Returns list of specific suggestions for improvement.
    """
    try:
        logger.info(f"Generating optimization suggestions for {resume.name}")
        
        suggestions = []
        
        # Analyze summary section
        if resume.summary:
            if len(resume.summary) < 50:
                suggestions.append(OptimizationSuggestion(
                    section="summary",
                    current_text=resume.summary,
                    suggested_text=resume.summary + " Add more context about achievements and career goals.",
                    reason="Summary should be more comprehensive (50+ characters recommended)"
                ))
        
        # Analyze experience descriptions
        for exp in resume.experience:
            if exp.description and len(exp.description) < 30:
                suggestions.append(OptimizationSuggestion(
                    section="experience",
                    current_text=exp.description,
                    suggested_text=exp.description + " Add quantifiable metrics and key accomplishments.",
                    reason="Experience descriptions should include metrics and specific achievements"
                ))
        
        # Analyze skills
        if len(resume.skills) < 5:
            suggestions.append(OptimizationSuggestion(
                section="skills",
                current_text=f"{len(resume.skills)} skills listed",
                suggested_text="Consider adding 5-10 relevant skills",
                reason="Most recruiters expect 5-10 relevant skills on a resume"
            ))
        
        logger.info(f"Generated {len(suggestions)} suggestions for {resume.name}")
        return suggestions
        
    except Exception as e:
        logger.error(f"Error optimizing resume: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error optimizing resume: {str(e)}"
        )


@app.get("/resume/{resume_id}", tags=["Resume Retrieval"])
async def get_resume(resume_id: str):
    """
    Retrieve a previously analyzed resume by ID.
    
    - **resume_id**: Unique resume identifier
    """
    logger.info(f"Retrieving resume {resume_id}")
    # This is a placeholder - implement actual storage retrieval
    return {
        "resume_id": resume_id,
        "message": "Resume retrieval not yet implemented"
    }


# ==================== Helper Functions ====================

def calculate_resume_score(resume: ResumeInput) -> float:
    """Calculate overall resume score based on various factors."""
    score = 50.0
    
    # Check for professional summary
    if resume.summary and len(resume.summary) > 30:
        score += 10
    
    # Check for contact information
    if resume.phone:
        score += 5
    
    # Evaluate skills
    if len(resume.skills) >= 5:
        score += 15
    elif len(resume.skills) >= 3:
        score += 10
    elif len(resume.skills) > 0:
        score += 5
    
    # Evaluate experience
    if len(resume.experience) >= 3:
        score += 20
    elif len(resume.experience) >= 1:
        score += 10
    
    # Evaluate education
    if len(resume.education) >= 1:
        score += 15
    
    return min(score, 100.0)


def identify_strengths(resume: ResumeInput) -> List[str]:
    """Identify resume strengths."""
    strengths = []
    
    if resume.summary and len(resume.summary) > 50:
        strengths.append("Well-written professional summary")
    
    if len(resume.skills) >= 5:
        strengths.append("Comprehensive skill set listed")
    
    if len(resume.experience) >= 2:
        strengths.append("Solid work experience history")
    
    if len(resume.education) >= 1:
        strengths.append("Educational background included")
    
    return strengths if strengths else ["Resume contains key sections"]


def identify_improvements(resume: ResumeInput) -> List[str]:
    """Identify areas for improvement."""
    improvements = []
    
    if not resume.summary:
        improvements.append("Add a professional summary")
    elif len(resume.summary) < 50:
        improvements.append("Expand professional summary for more impact")
    
    if len(resume.skills) < 5:
        improvements.append("Add more relevant skills")
    
    if len(resume.experience) == 0:
        improvements.append("Include work experience")
    
    if not all(exp.description for exp in resume.experience):
        improvements.append("Add descriptions to all experience entries")
    
    return improvements


def generate_recommendations(resume: ResumeInput) -> List[str]:
    """Generate specific recommendations."""
    recommendations = []
    
    recommendations.append("Use action verbs in job descriptions (e.g., Led, Implemented, Designed)")
    recommendations.append("Quantify achievements with metrics and numbers")
    recommendations.append("Tailor resume content for specific job applications")
    recommendations.append("Ensure consistent formatting and professional appearance")
    recommendations.append("Optimize for ATS (Applicant Tracking Systems) by using standard keywords")
    
    return recommendations


# ==================== Entry Point ====================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
