"""
Manages chat rooms and message history between Siva and Hari
FIXED: Added global instance
"""
import time
from datetime import datetime

class ChatManager:
    def __init__(self):
        self.rooms = {}
        self.chat_histories = {}
    
    def join_room(self, room_id, user_type, session_id):
        """Add user to a chat room"""
        if room_id not in self.rooms:
            self.rooms[room_id] = {}
            self.chat_histories[room_id] = []
        
        self.rooms[room_id][session_id] = user_type
        
        # Add join message to history
        join_message = {
            'id': len(self.chat_histories[room_id]),
            'from': 'system',
            'message': f"{user_type.capitalize()} joined the chat",
            'type': 'system',
            'timestamp': datetime.now().isoformat()
        }
        self.chat_histories[room_id].append(join_message)
        return True
    
    def leave_room(self, room_id, user_type):
        """Remove user from chat room"""
        if room_id in self.rooms:
            # Find and remove the user's session
            sessions_to_remove = []
            for session_id, ut in self.rooms[room_id].items():
                if ut == user_type:
                    sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del self.rooms[room_id][session_id]
            
            # Add leave message to history
            leave_message = {
                'id': len(self.chat_histories[room_id]),
                'from': 'system',
                'message': f"{user_type.capitalize()} left the chat",
                'type': 'system',
                'timestamp': datetime.now().isoformat()
            }
            self.chat_histories[room_id].append(leave_message)
    
    def add_message(self, room_id, user_type, message, message_type="text"):
        """Add a message to chat history"""
        if room_id not in self.chat_histories:
            self.chat_histories[room_id] = []
        
        chat_message = {
            'id': len(self.chat_histories[room_id]),
            'from': user_type,
            'message': message,
            'type': message_type,
            'timestamp': datetime.now().isoformat()
        }
        
        self.chat_histories[room_id].append(chat_message)
        return chat_message
    
    def update_last_message(self, room_id, new_message):
        """Update the last message in chat history"""
        if (room_id in self.chat_histories and 
            self.chat_histories[room_id]):
            self.chat_histories[room_id][-1]['message'] = new_message
    
    def get_chat_history(self, room_id):
        """Get all messages from a room"""
        return self.chat_histories.get(room_id, [])
    
    def get_room_sessions(self, room_id):
        """Get all sessions in a room"""
        return self.rooms.get(room_id, {})
    
    def get_user_type(self, room_id, session_id):
        """Get user type for a session in a room"""
        if room_id in self.rooms:
            return self.rooms[room_id].get(session_id)
        return None

# Global instance
chat_manager = ChatManager()