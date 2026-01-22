import unittest
import torch
from src.models.unet import ConditionalUNet
from src.models.scheduler import DiffusionScheduler

class TestConditionalUNet(unittest.TestCase):
    def setUp(self):
        self.model = ConditionalUNet()
        self.scheduler = DiffusionScheduler(steps=1000)
        self.batch_size = 4
        self.input_shape = (self.batch_size, 128, 1)
        self.radar_cond_shape = (self.batch_size, 128, 1)
        self.time_steps = torch.randint(0, 1000, (self.batch_size,))

    def test_forward_pass(self):
        x_t = torch.randn(self.input_shape)
        radar_cond = torch.randn(self.radar_cond_shape)
        output = self.model(x_t, self.time_steps, radar_cond)
        self.assertEqual(output.shape, (self.batch_size, 128, 1), "Output shape mismatch")

    def test_model_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertGreater(total_params, 0, "Model has no parameters")

class TestDiffusionScheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = DiffusionScheduler(steps=1000)

    def test_q_sample(self):
        x_0 = torch.randn((4, 128, 1))
        noise = torch.randn((4, 128, 1))
        t = torch.randint(0, 1000, (4,))
        x_t = self.scheduler.q_sample(x_0, t, noise)
        self.assertEqual(x_t.shape, x_0.shape, "q_sample output shape mismatch")

if __name__ == '__main__':
    unittest.main()