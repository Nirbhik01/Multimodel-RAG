import os
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class MongoDBClient:
    def __init__(self):
        connection_string = os.getenv('MONGO_DB_CONNECTION_STRING')
        if not connection_string:
            raise ValueError("MONGO_DB_CONNECTION_STRING not found in environment variables")
        
        self.client = MongoClient(connection_string)
        self.db = self.client['multimodal_rag']
        self.conversations = self.db['conversations']

    def create_conversation(self, title="New Chat"):
        conversation = {
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "messages": []
        }
        result = self.conversations.insert_one(conversation)
        return str(result.inserted_id)

    def get_conversations(self):
        conversations = list(self.conversations.find({}, {"messages": 0}).sort("updated_at", -1))
        for conv in conversations:
            conv["_id"] = str(conv["_id"])
        return conversations

    def get_conversation(self, conversation_id):
        try:
            conv = self.conversations.find_one({"_id": ObjectId(conversation_id)})
            if conv:
                conv["_id"] = str(conv["_id"])
            return conv
        except Exception:
            return None

    def add_message(self, conversation_id, message):
        try:
            self.conversations.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"messages": message},
                    "$set": {"updated_at": datetime.utcnow()}
                }
            )
            return True
        except Exception:
            return False

    def update_conversation_title(self, conversation_id, title):
        try:
            self.conversations.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"title": title, "updated_at": datetime.utcnow()}}
            )
            return True
        except Exception:
            return False

    def delete_conversation(self, conversation_id):
        try:
            self.conversations.delete_one({"_id": ObjectId(conversation_id)})
            return True
        except Exception:
            return False

# Singleton instance
db_client = MongoDBClient()
