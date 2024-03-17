import asyncio
import unittest
from unittest.mock import Mock, MagicMock
from actions.actions import ActionHandleBankTransfer  # Replace with the actual path to your actions file
import asyncio
import unittest
import mysql.connector
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
import mysql.connector

class TestActionHandleBankTransfer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Use existing database with initial values
        cls.connection = mysql.connector.connect(
            user='root',
            password='Czw200513',
            host='localhost',
            database='banking_chatbot',  # Use your actual database
            autocommit=True
        )
        cls.cursor = cls.connection.cursor()

    @classmethod
    def tearDownClass(cls):
        # Clean up and close connections after tests
        cls.cursor.close()
        cls.connection.close()

    def reset_account_balances(self):
        # Reset account balances to the initial state
        initial_balances = {'Bob': 2000, 'Alice': 1500, 'Clare': 500, 'David': 700}
        for account, balance in initial_balances.items():
            self.cursor.execute(
                "UPDATE Accounts SET AccountBalance = %s WHERE AccountName = %s",
                (balance, account)
            )
    def test_action_handle_bank_transfer(self):

        self.reset_account_balances()# Ensure initial state before tests
        # Provided test cases
        test_cases = [
            {"amount": 100, "recipient": "Alice"},
            {"amount": 200, "recipient": "Clare"},
            {"amount": 20, "recipient": "David"},
            {"amount": 50, "recipient": "Alice"},
            {"amount": 80, "recipient": "Clare"},
            {"amount": 80, "recipient": "David"},
            {"amount": 30, "recipient": "Alice"},
            {"amount": 20, "recipient": "Clare"},
            {"amount": 100, "recipient": "David"},
            {"amount": 40, "recipient": "Alice"},
            {"amount": 100, "recipient": "Clare"},
            {"amount": 60, "recipient": "David"},
            {"amount": 80, "recipient": "Alice"},
            {"amount": 60, "recipient": "Clare"},
            {"amount": 70, "recipient": "David"},
            {"amount": 100, "recipient": "Alice"},
            {"amount": 40, "recipient": "Clare"},
            {"amount": 70, "recipient": "David"},
            {"amount": 10, "recipient": "Alice"},
            {"amount": 20, "recipient": "Clare"},
            {"amount": 10, "recipient": "David"},
            {"amount": 30, "recipient": "Alice"},
            {"amount": 20, "recipient": "Clare"},
            {"amount": 30, "recipient": "David"},
            {"amount": 5, "recipient": "Alice"},
            {"amount": 10, "recipient": "Clare"},
            {"amount": 5, "recipient": "David"},
            {"amount": 15, "recipient": "Alice"},
            {"amount": 10, "recipient": "Clare"},
            {"amount": 15, "recipient": "David"},
            {"amount": 8, "recipient": "Alice"},
            {"amount": 12, "recipient": "Clare"},
            {"amount": 14, "recipient": "David"},
            {"amount": 13, "recipient": "Alice"},
            {"amount": 11, "recipient": "Clare"},
            {"amount": 10, "recipient": "David"},
            {"amount": 9, "recipient": "Alice"},
            {"amount": 7, "recipient": "Clare"},
            {"amount": 6, "recipient": "David"},
            {"amount": 10, "recipient": "Alice"},
            {"amount": 10, "recipient": "Clare"},
            {"amount": 10, "recipient": "David"},
        ]


        # Create dispatcher and tracker
        dispatcher = CollectingDispatcher()
        domain = {}

        for case in test_cases:
            # Create a new tracker for each test case
            tracker = Tracker(
                sender_id='default',
                slots={'amount': case['amount'], 'recipient': case['recipient']},
                latest_message={},
                events=[],
                paused=False,
                followup_action=None,
                active_loop=None,
                latest_action_name=None
            )

            # Run the action
            action = ActionHandleBankTransfer()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(action.run(dispatcher, tracker, domain))

        # After running all test cases, verify the final balances
        account_balances = {
            "Bob": 400,
            "Alice": 2000,
            "Clare": 1100,
            "David": 1200,
        }

        for account_name, expected_balance in account_balances.items():
            with self.subTest(account_name=account_name):
                self.cursor.execute("SELECT AccountBalance FROM Accounts WHERE AccountName = %s", (account_name,))
                actual_balance = self.cursor.fetchone()[0]
                self.assertEqual(actual_balance, expected_balance, f"Balance for {account_name} does not match expected value.")

if __name__ == '__main__':
    unittest.main()

