# tests/test_metachor.py
import unittest
from unittest import TestCase, mock
import time
from metachor.types import Phase, ResourceConstraints, Message
from metachor.voice import Voice
from metachor.ensemble import Ensemble

class TestVoice(TestCase):
    def setUp(self):
        self.voice = Voice("test-model", "Test system prompt")
    
    def test_send_basic_message(self):
        response = self.voice.send(
            content="Test message",
            to_model="other-model",
            phase=Phase.INITIALIZATION
        )
        self.assertIsInstance(response, Message)
        self.assertEqual(response.from_model, "test-model")
        self.assertEqual(response.to_model, "other-model")
        self.assertEqual(response.phase, Phase.INITIALIZATION)
    
    def test_prepare_prompt(self):
        context = [
            Message("Previous message", 10, "model1", "model2", Phase.INITIALIZATION)
        ]
        prompt = self.voice.prepare_prompt("Current message", context)
        self.assertIn("System: Test system prompt", prompt)
        self.assertIn("Previous message", prompt)
        self.assertIn("Current message", prompt)

    def test_history_management(self):
        self.voice.conversation_history.append(
            Message("Test", 5, "test-model", "other-model", Phase.INITIALIZATION)
        )
        self.assertEqual(len(self.voice.conversation_history), 1)
        self.voice.forget_history()
        self.assertEqual(len(self.voice.conversation_history), 0)

class TestEnsemble(TestCase):
    def setUp(self):
        self.voice1 = Voice("model1", "Test prompt 1")
        self.voice2 = Voice("model2", "Test prompt 2")
        self.ensemble = Ensemble([self.voice1, self.voice2])
        self.constraints = ResourceConstraints(
            max_tokens=1000,
            max_iterations=10,
            max_time=30.0
        )
    
    def test_initialization(self):
        self.assertEqual(len(self.ensemble.voices), 2)
        self.assertEqual(self.ensemble.voices[0].model_id, "model1")
        self.assertEqual(self.ensemble.voices[1].model_id, "model2")
    
    def test_voice_rotation(self):
        next_voice = self.ensemble._get_next_voice(self.voice1)
        self.assertEqual(next_voice.model_id, "model2")
        next_voice = self.ensemble._get_next_voice(self.voice2)
        self.assertEqual(next_voice.model_id, "model1")
    
    @mock.patch.object(Voice, 'send')
    def test_meta_discussion(self, mock_send):
        mock_send.return_value = Message(
            "Test response", 10, "model1", "model2", Phase.INITIALIZATION
        )
        self.ensemble.initialize_meta_discussion(iterations=1)
        # Should call send twice - once for each voice
        self.assertEqual(mock_send.call_count, 2)
    
    @mock.patch.object(Voice, 'send')
    def test_basic_response_generation(self, mock_send):
        mock_send.return_value = Message(
            "Test response", 10, "model1", "model2", Phase.USER_ANALYSIS
        )
        response = self.ensemble.send("Test input", self.constraints)
        self.assertTrue(mock_send.called)
        self.assertIsInstance(response, str)

    def test_response_integration(self):
        msg = Message("New content", 5, "model1", "model2", Phase.RESPONSE_DRAFTING)
        # Test empty current
        result = self.ensemble._integrate_response("", msg)
        self.assertEqual(result, "New content")
        # Test with existing content
        result = self.ensemble._integrate_response("Existing content", msg)
        self.assertIn("Existing content", result)
        self.assertIn("New content", result)

class TestEndToEnd(TestCase):
    def test_simple_conversation(self):
        """Test a simple conversation flow through the system."""
        voice1 = Voice("model1", "System prompt 1")
        voice2 = Voice("model2", "System prompt 2")
        ensemble = Ensemble([voice1, voice2])
        
        constraints = ResourceConstraints(
            max_tokens=100,
            max_iterations=5,
            max_time=10.0
        )
        
        # Should complete without errors
        response = ensemble.send("Test question", constraints)
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_resource_constraints(self):
        """Test that resource constraints are respected."""
        voice1 = Voice("model1")
        ensemble = Ensemble([voice1])
        
        # Set very restrictive constraints
        constraints = ResourceConstraints(
            max_tokens=10,
            max_iterations=1,
            max_time=1.0
        )
        
        start_time = time.time()
        response = ensemble.send("Test input", constraints)
        elapsed_time = time.time() - start_time
        
        self.assertLess(elapsed_time, constraints.max_time + 0.1)  # Allow small buffer
        self.assertLess(len(response.split()), constraints.max_tokens)

if __name__ == '__main__':
    unittest.main()
