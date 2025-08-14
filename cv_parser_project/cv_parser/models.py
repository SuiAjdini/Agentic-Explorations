from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class WorkExperience(BaseModel):
    """Data model for a single work experience."""
    job_title: str = Field(description="The job title or position.")
    company: str = Field(description="The name of the company.")
    location: Optional[str] = Field(description="The location of the company (e.g., city, state).")
    start_date: Optional[str] = Field(description="The start date of the employment.")
    end_date: Optional[str] = Field(description="The end date of the employment. Can be 'Present'.")
    description: Optional[str] = Field(description="A brief description of the role and responsibilities.")

class Education(BaseModel):
    """Data model for a single education entry."""
    institution: str = Field(description="The name of the educational institution.")
    degree: str = Field(description="The degree obtained (e.g., Bachelor of Science).")
    field_of_study: Optional[str] = Field(description="The field of study (e.g., Computer Science).")
    graduation_date: Optional[str] = Field(description="The date of graduation.")

class CandidateProfile(BaseModel):
    """The main data model for the extracted CV information."""
    name: str = Field(description="The full name of the candidate.")
    email: Optional[str] = Field(description="The email address of the candidate.")
    phone: Optional[str] = Field(description="The phone number of the candidate.")
    summary: Optional[str] = Field(description="A brief summary or objective from the CV.")
    skills: List[str] = Field(description="A list of skills mentioned in the CV.")
    work_experience: List[WorkExperience] = Field(description="A list of the candidate's work experiences.")
    education: List[Education] = Field(description="A list of the candidate's educational background.")

class FeedbackIntent(BaseModel):
    """Data model for classifying the user's feedback intent."""
    intent: Literal["confirm", "correct", "query", "unclear"] = Field(
        description="The user's intent. Must be 'confirm' for approval, 'correct' for fixing errors, 'query' for asking questions about the data, or 'unclear'."
    )
    details: Optional[str] = Field(description="If intent is 'query' or 'correct', this contains the user's specific request or the identified error.")