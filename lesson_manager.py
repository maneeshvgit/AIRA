# lesson_manager.py
from agent import ask_agent
import uuid

class LessonManager:
    def __init__(self, thread_id=None):
        if thread_id is None:
            thread_id = f"classroom_session_{uuid.uuid4().hex[:8]}"
        self.thread_id = thread_id
        self.lesson_buffer = []       
        self.current_line_idx = 0     
        self.paused_context = []     

    def start_lesson(self, topic: str):
        
        ai_response = ask_agent(topic, thread_id=self.thread_id)
        self.lesson_buffer = self._split_into_chunks(ai_response)
        self.current_line_idx = 0

    def _split_into_chunks(self, text: str):
        
        # Simple split by ". " but keep punctuation; later you could improve with nltk
        chunks = [chunk.strip() for chunk in text.replace("\n", " ").split(". ") if chunk.strip()]
        return chunks

    def get_next_line(self):
        
        if self.current_line_idx < len(self.lesson_buffer):
            line = self.lesson_buffer[self.current_line_idx]
            self.current_line_idx += 1
            return line
        return None

    def pause_for_question(self, question: str):
        """
        Handle a student interruption/question and update the lesson flow to resume naturally.
        """
        # Store a recap context (last 2-3 lines spoken)
        start_idx = max(0, self.current_line_idx - 3)
        self.paused_context = self.lesson_buffer[start_idx:self.current_line_idx]

        recap_context = " ".join(self.paused_context)

        # Ask LLM to answer question and resume lesson
        followup_prompt = (
            f"We paused the lesson here:\n'{recap_context}'.\n"
            f"A student asked: '{question}'.\n"
            f"Please answer the question clearly and then smoothly continue the lesson from this point, "
            f"with a brief natural recap before continuing."
        )

        ai_followup = ask_agent(followup_prompt, thread_id=self.thread_id)
        
        # Replace the lesson buffer with the new continuation from LLM
        self.lesson_buffer = self._split_into_chunks(ai_followup)
        self.current_line_idx = 0

    def is_lesson_done(self):
        """
        Check if the current lesson content is exhausted.
        """
        return self.current_line_idx >= len(self.lesson_buffer)
