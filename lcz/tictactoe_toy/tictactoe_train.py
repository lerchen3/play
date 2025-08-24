import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tictactoe_interface import TicTacToeGame, MCTS
from tictactoe_net import TicTacToeNet

GAMES = 50
ITERATIONS = 40
EPOCHS = 10
BATCH_SIZE = 8

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

net = TicTacToeNet().to(DEVICE)
optimizer = optim.Adam(net.parameters(), lr=0.01)

for it in range(ITERATIONS):
    data = []
    print(f"Iteration {it+1}/{ITERATIONS}")
    
    for g in range(GAMES):
        game = TicTacToeGame()
        mcts = MCTS(net, device=DEVICE)
        
        states = []
        policies = []
        players = []
        
        while not game.is_game_over():
            state = game.tensor_state().to(DEVICE)
            policy = mcts.search(game)
            
            states.append(state.cpu())
            policies.append(torch.from_numpy(policy))
            players.append(game.turn())
            
            # Sample move from policy
            if policy.sum() > 0:
                move = np.random.choice(9, p=policy)
            else:
                # raise an error! the policy should sum to 1!
                raise ValueError("Policy should sum to 1!")
            
            game.move(move)
        
        # Get final result
        result = game.result()
        
        # Convert to training data
        for i, (state, policy, player) in enumerate(zip(states, policies, players)):
            # Assign value based on game result and player perspective
            if result == 0:  # draw
                value = 0.0
            elif result == player:  # win for this player
                value = 1.0
            else:  # loss for this player
                value = -1.0
                
            data.append((state, policy, value))
    
    print(f"Collected {len(data)} training samples")
    
    # Training
    if data:
        for epoch in range(EPOCHS):
            random.shuffle(data)
            total_loss = 0.0
            batches = 0
            
            for i in range(0, len(data), BATCH_SIZE):
                batch = data[i:i+BATCH_SIZE]
                if len(batch) < BATCH_SIZE:
                    continue
                    
                states = torch.stack([item[0] for item in batch]).to(DEVICE)
                target_policies = torch.stack([item[1] for item in batch]).to(DEVICE)
                target_values = torch.tensor([item[2] for item in batch]).to(DEVICE)
                
                optimizer.zero_grad()
                
                pred_policies, pred_values = net(states)
                
                # Policy loss (cross entropy)
                policy_loss = -torch.mean(torch.sum(target_policies * torch.log_softmax(pred_policies, dim=1), dim=1))
                
                # Value loss (MSE)
                value_loss = torch.mean((target_values - pred_values.squeeze()) ** 2)
                
                total_loss_batch = policy_loss + value_loss
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                batches += 1
            
            if batches > 0:
                avg_loss = total_loss / batches
                print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")
    
    # Save model
    torch.save(net.state_dict(), f'tictactoe_model_iter_{it+1}.pt')

torch.save(net.state_dict(), 'tictactoe_model.pt')
print("Training complete!")
