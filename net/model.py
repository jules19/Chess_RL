"""
Policy-Value Neural Network - AlphaZero Architecture

This module implements a ResNet-based neural network for chess that outputs:
1. Policy: Probability distribution over moves
2. Value: Position evaluation (-1 to +1)

Architecture:
    Input (20, 8, 8)
    → ResNet Trunk (4-6 blocks)
    → Policy Head (4672-dim move probabilities)
    → Value Head (scalar position evaluation)

Start small (4 ResBlocks, 128 channels) for fast iteration on Mac mini M4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection (building block of ResNets).

    Architecture:
        x → Conv → BatchNorm → ReLU → Conv → BatchNorm → (+x) → ReLU

    The skip connection (+x) allows gradients to flow directly through
    the network, enabling training of very deep networks.

    Args:
        channels: Number of convolutional channels
    """

    def __init__(self, channels=128):
        super(ResidualBlock, self).__init__()

        # First conv layer
        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,  # Same padding (output size = input size)
            bias=False  # BatchNorm has bias, so conv doesn't need it
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Second conv layer
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        """
        Forward pass with skip connection.

        Args:
            x: Input tensor (batch_size, channels, 8, 8)

        Returns:
            Output tensor (batch_size, channels, 8, 8)
        """
        # Save input for skip connection
        identity = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection (residual)
        out = out + identity

        # Final activation
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """
    Policy head: Outputs probability distribution over all possible moves.

    Architecture:
        (batch, 128, 8, 8) → Conv 1x1 (reduce to 2 channels)
                           → BatchNorm → ReLU
                           → Flatten
                           → Linear → 4672 logits

    The 4672 dimensions correspond to all possible chess moves:
    - 64 source squares × 73 possible destinations per square
    - (simplified: we'll use a subset or mask illegal moves)

    For now, we use a simple 4672-dim output and mask illegal moves.
    """

    def __init__(self, in_channels=128, num_moves=4672):
        super(PolicyHead, self).__init__()

        # Reduce channels with 1x1 convolution
        self.conv = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm2d(2)

        # Fully connected to move space
        # After conv: (batch, 2, 8, 8) → flatten to (batch, 128)
        self.fc = nn.Linear(2 * 8 * 8, num_moves)

    def forward(self, x):
        """
        Forward pass through policy head.

        Args:
            x: Feature maps from trunk (batch_size, 128, 8, 8)

        Returns:
            Policy logits (batch_size, 4672) - unnormalized log probabilities
        """
        # Reduce channels
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Flatten
        out = out.view(out.size(0), -1)  # (batch, 2*8*8)

        # Fully connected to move space
        policy_logits = self.fc(out)  # (batch, 4672)

        return policy_logits


class ValueHead(nn.Module):
    """
    Value head: Outputs scalar position evaluation.

    Architecture:
        (batch, 128, 8, 8) → Conv 1x1 (reduce to 1 channel)
                           → BatchNorm → ReLU
                           → Flatten
                           → Linear(64, 256) → ReLU
                           → Linear(256, 1) → Tanh
                           → Scalar in [-1, 1]

    Output interpretation:
        +1.0 = White is winning
         0.0 = Equal position / draw
        -1.0 = Black is winning
    """

    def __init__(self, in_channels=128, hidden_size=256):
        super(ValueHead, self).__init__()

        # Reduce channels with 1x1 convolution
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm2d(1)

        # Fully connected layers
        # After conv: (batch, 1, 8, 8) → flatten to (batch, 64)
        self.fc1 = nn.Linear(1 * 8 * 8, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        Forward pass through value head.

        Args:
            x: Feature maps from trunk (batch_size, 128, 8, 8)

        Returns:
            Value scalar (batch_size, 1) in range [-1, 1]
        """
        # Reduce channels
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Flatten
        out = out.view(out.size(0), -1)  # (batch, 64)

        # Hidden layer
        out = self.fc1(out)
        out = F.relu(out)

        # Output layer
        value = self.fc2(out)  # (batch, 1)
        value = torch.tanh(value)  # Squash to [-1, 1]

        return value


class PolicyValueNetwork(nn.Module):
    """
    Complete policy-value network combining shared trunk with two heads.

    This is the AlphaZero architecture:
    - Shared convolutional trunk learns board features
    - Policy head predicts good moves
    - Value head predicts position evaluation

    Args:
        input_channels: Number of input planes (default: 20)
            - 12 piece planes (6 types × 2 colors)
            - 2 repetition counters
            - 1 side to move
            - 4 castling rights
            - 1 fifty-move counter
        num_res_blocks: Number of residual blocks (default: 4)
            Start with 4 for fast iteration, can scale to 20-40
        num_channels: Channels in residual blocks (default: 128)
            Start with 128, can scale to 256-512
    """

    def __init__(self, input_channels=20, num_res_blocks=4, num_channels=128):
        super(PolicyValueNetwork, self).__init__()

        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels

        # Initial convolution (expand from input channels to num_channels)
        self.conv_input = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_channels)

        # Residual tower (shared trunk)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # Policy head
        self.policy_head = PolicyHead(in_channels=num_channels)

        # Value head
        self.value_head = ValueHead(in_channels=num_channels)

    def forward(self, x):
        """
        Forward pass through the entire network.

        Args:
            x: Input tensor (batch_size, 20, 8, 8)
               Board representation with 20 feature planes

        Returns:
            policy_logits: (batch_size, 4672) - unnormalized log probabilities
            value: (batch_size, 1) - position evaluation in [-1, 1]
        """
        # Initial convolution
        out = self.conv_input(x)
        out = self.bn_input(out)
        out = F.relu(out)

        # Residual tower
        for res_block in self.res_blocks:
            out = res_block(out)

        # Policy head
        policy_logits = self.policy_head(out)

        # Value head
        value = self.value_head(out)

        return policy_logits, value

    def get_policy_value(self, x, legal_moves_mask=None):
        """
        Get policy and value with optional legal move masking.

        This is the main inference function used during MCTS.

        Args:
            x: Input tensor (batch_size, 20, 8, 8)
            legal_moves_mask: Optional binary mask (batch_size, 4672)
                             1 = legal move, 0 = illegal move

        Returns:
            policy_probs: (batch_size, 4672) - normalized probabilities
            value: (batch_size, 1) - position evaluation
        """
        # Forward pass
        policy_logits, value = self.forward(x)

        # Mask illegal moves (set to large negative value)
        if legal_moves_mask is not None:
            policy_logits = policy_logits + (legal_moves_mask - 1.0) * 1e9

        # Convert logits to probabilities
        policy_probs = F.softmax(policy_logits, dim=1)

        return policy_probs, value

    def num_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        print("=" * 70)
        print("Policy-Value Network Summary")
        print("=" * 70)
        print(f"Input channels: {self.input_channels}")
        print(f"Residual blocks: {self.num_res_blocks}")
        print(f"Channels per block: {self.num_channels}")
        print(f"Total parameters: {self.num_parameters():,}")
        print(f"Model size: ~{self.num_parameters() * 4 / 1024 / 1024:.1f} MB (FP32)")
        print("=" * 70)


def create_model(num_res_blocks=4, num_channels=128):
    """
    Factory function to create a policy-value network.

    Args:
        num_res_blocks: Number of residual blocks (default: 4)
        num_channels: Channels in residual blocks (default: 128)

    Returns:
        PolicyValueNetwork instance
    """
    model = PolicyValueNetwork(
        input_channels=20,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels
    )
    return model


if __name__ == "__main__":
    """Quick test of the network architecture."""
    print("Testing Policy-Value Network Architecture...")
    print()

    # Create model
    model = create_model(num_res_blocks=4, num_channels=128)
    model.summary()
    print()

    # Test with dummy input
    print("Testing forward pass with dummy input...")
    batch_size = 2
    dummy_input = torch.randn(batch_size, 20, 8, 8)

    # Forward pass
    policy_logits, value = model(dummy_input)

    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Policy logits shape: {policy_logits.shape}")
    print(f"✓ Value shape: {value.shape}")
    print()

    # Validate shapes
    assert policy_logits.shape == (batch_size, 4672), f"Policy shape wrong: {policy_logits.shape}"
    assert value.shape == (batch_size, 1), f"Value shape wrong: {value.shape}"
    print("✅ Shape validation PASSED!")
    print()

    # Test value range
    print("Testing value range...")
    print(f"✓ Value output: {value.squeeze().tolist()}")
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0), "Value outside [-1, 1] range!"
    print("✅ Value range validation PASSED (all values in [-1, 1])!")
    print()

    # Test legal move masking
    print("Testing legal move masking...")
    legal_mask = torch.zeros(batch_size, 4672)
    legal_mask[:, :20] = 1.0  # Only first 20 moves are legal

    policy_probs, value = model.get_policy_value(dummy_input, legal_mask)
    print(f"✓ Policy probabilities shape: {policy_probs.shape}")
    print(f"✓ Sum of probabilities: {policy_probs.sum(dim=1).tolist()}")
    print(f"✓ Probability mass on illegal moves: {policy_probs[:, 20:].sum(dim=1).tolist()}")

    assert torch.allclose(policy_probs.sum(dim=1), torch.ones(batch_size), atol=1e-5), "Probabilities don't sum to 1!"
    assert torch.all(policy_probs[:, 20:] < 1e-6), "Illegal moves have non-zero probability!"
    print("✅ Legal move masking PASSED!")
    print()

    print("=" * 70)
    print("✅ ALL TESTS PASSED! Network architecture is ready.")
    print("=" * 70)
