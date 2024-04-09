# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import actions.ActionsGeneralInfo as ActionsGeneralInfo
from typing import Any, Text, Dict
from rasa_sdk.events import SlotSet, FollowupAction
from rasa_sdk import Action, Tracker
import openai
from typing import List
from rasa_sdk.executor import CollectingDispatcher
from datetime import datetime
import mysql.connector

# Set API Key
os.environ['OPENAI_API_KEY'] = 'YOUR-KEY'
openai.api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model='gpt-4',temperature = 0.1)

class ActionClassifyIntent(Action):
    def name(self) -> Text:
        return "action_classify_intent"

    def run(self, dispatcher, tracker, domain):

        message_text = tracker.latest_message.get('text')
        print(message_text)
        prompt = f"Classify the following user message into one of two intents - 'general_info' or 'bank_transfer': '{message_text}'. Which intent does it belong to? You should only return the intent name"

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        intent_name = response.choices[0].message.content.strip()

        if intent_name not in ["general_info", "bank_transfer"]:
            intent_name = "general_info"  # Default to general_info if unsure

        # Set the appropriate follow-up action based on the classified intent
        if intent_name == "general_info":
            followup_action = "action_respond_general_info"
        elif intent_name == "bank_transfer":
            followup_action = "action_handle_bank_transfer"

        # Return the FollowupAction event along with any other events
        return [SlotSet("predicted_intent", intent_name), FollowupAction(followup_action)]

class ActionRespondGeneralInfo(Action):
    def name(self) -> Text:
        return "action_respond_general_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:


        user_message = tracker.latest_message.get('text')
        txt = {'role': 'user', 'content': f"{user_message}"}

        context=[]

        for dialogue in ActionsGeneralInfo.dialogues:
            for message in dialogue["messages"]:
                context.append(message)

        prompt = [ {'role':'assistant', 'content':"""
        You are a chatbot designed for an academic banking scenario. Your task is to respond to customer queries in a concise manner, 
        using up to three sentences, without involving real-life security or privacy concerns.
        You have basic hypothetical information about the bank for this customer, including his recent transaction details.
        As his banking assistant, please provide simple and clear answers about basic customer information. 
        Responses should be as accurate as possible and limited to 50 words.
        """} ]
        context.append(txt)

        # Add the new input to the prompt
        prompt.append({'user': f"{user_message}\n assistant:"})
        if ActionsGeneralInfo.is_question_relevant(user_message,ActionsGeneralInfo.keywords,ActionsGeneralInfo.common_greetings):
            # Get the assistant's response
            retriever, memory = ActionsGeneralInfo.qaRetrival()
            qa = ConversationalRetrievalChain.from_llm(
                llm,
                retriever=retriever,
                memory=memory,
            )
            response = qa({"question": prompt, "chat_history": context})
            answer = response["answer"]
        else:
            answer = "Sorry，I can only answer the question relevant to Bank."

        txt = {'role': 'Banking assistant', 'content': f"{answer}"}
        context.append(txt)

        dispatcher.utter_message(text=answer+" Is there anything else I can assist you with?")

        return []


class ActionConfirmTransfer(Action):
    def name(self) -> Text:
        return "action_confirm_transfer"

    def run(
            self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]
    ) -> List[Dict]:

        amount = tracker.get_slot("amount")
        recipient = tracker.get_slot("recipient")

        confirm_message = f"Please confirm: Transfer {amount} to {recipient}"
        buttons = [{"title": "Yes", "payload": "/confirm_yes"}, {"title": "No", "payload": "/confirm_no"}]
        dispatcher.utter_message(text=confirm_message, buttons=buttons)

        return []

class ActionHandleBankTransfer(Action):
    def name(self) -> Text:
        return "action_handle_bank_transfer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Get the slot values for the amount and recipient
        amount = float(tracker.get_slot("amount"))
        recipient = tracker.get_slot("recipient")

        # Database connection parameters
        config = {
            'user': 'root',
            'password': 'Czw200513',
            'host': 'localhost',
            'database': 'banking_chatbot',
        }

        # Connect to the database
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()

        try:
            # Update sender's account balance (Bob)
            cursor.execute(
                "UPDATE Accounts SET AccountBalance = AccountBalance - %s WHERE AccountName = 'Bob'",
                (amount,)
            )

            # Update recipient's account balance (Alice)
            cursor.execute(
                "UPDATE Accounts SET AccountBalance = AccountBalance + %s WHERE AccountName = %s",
                (amount, recipient)
            )

            # Get the new balance of Bob's account for the transaction record
            cursor.execute("SELECT AccountBalance FROM Accounts WHERE AccountName = 'Bob'")
            new_balance = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM transactions")
            number_of_transactions = cursor.fetchone()[0]

            # Insert a new transaction
            transaction_number = number_of_transactions + 1
            cursor.execute(
                "INSERT INTO transactions (transaction_number, date, amount_change, balance_after, receiver, information) VALUES (%s, %s, %s, %s, %s, %s)",
                (transaction_number, datetime.today().strftime('%Y-%m-%d'), -amount, new_balance, recipient, f"Bank transfer to {recipient}")
            )

            # Commit the changes
            connection.commit()

            dispatcher.utter_message(text="Money transfer succeeded! Your new account balance is "+ str(new_balance)+", Is there anything else I can assist you with?" )

        except mysql.connector.Error as err:
            # Handle errors and rollback changes
            connection.rollback()
            dispatcher.utter_message(text="An error occurred: {}".format(err))
            # You might want to log this error.

        finally:
            # Close the connection
            cursor.close()
            connection.close()

        return []
