import streamlit as st
import networkx as nx
import alpaca_trade_api as tradeapi
import pandas as pd
import requests
import json
import csv
import time
from datetime import datetime
from itertools import permutations
import plotly.graph_objects as go

class CryptoArbitrageApp:
    def __init__(self):
        # Constants
        self.COINS = [
            ("bitcoin", "btc"), ("ethereum", "eth"), ("ripple", "xrp"),
            ("bnb", "bnb"), ("bitcoin-cash", "bch"), ("eos", "eos"),
            ("litecoin", "ltc"), ("maker", "mkr"), ("chainlink", "link"),
            ("aave", "aave"), ("dai", "dai"), ("dogecoin", "doge"),
            ("shiba-inu", "shib")
        ]
        self.COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3/simple/price"
        self.graph = nx.DiGraph()

    def initialize_alpaca(self, api_key, api_secret, base_url):
        """Initialize Alpaca API connection"""
        self.api = tradeapi.REST(api_key, api_secret, base_url=base_url)
        return self.api.get_account()

    def fetch_exchange_rates(self):
        """Fetch current exchange rates from CoinGecko"""
        exchange_rates = {}
        for coin1, coin2 in permutations(self.COINS, 2):
            params = {
                'ids': f"{coin1[0]},{coin2[0]}",
                'vs_currencies': f"{coin1[1]},{coin2[1]}"
            }
            try:
                response = requests.get(self.COINGECKO_BASE_URL, params=params)
                data = response.json()
                
                if coin1[0] in data and coin2[1] in data[coin1[0]]:
                    rate = data[coin1[0]][coin2[1]]
                    exchange_rates[(coin1[1], coin2[1])] = rate
                
                time.sleep(1)  # Rate limiting
            except Exception as e:
                st.error(f"Error fetching rates for {coin1[0]}-{coin2[0]}: {str(e)}")
        
        return exchange_rates

    def update_graph(self, exchange_rates):
        """Update graph with current exchange rates"""
        self.graph.clear()
        for (from_coin, to_coin), rate in exchange_rates.items():
            self.graph.add_weighted_edges_from([(from_coin, to_coin, rate)])

    def find_arbitrage_opportunities(self, min_profit_threshold=1.001):
        """Find arbitrage opportunities in the graph"""
        opportunities = []
        
        for coin1, coin2 in permutations(self.COINS, 2):
            try:
                paths = list(nx.all_simple_paths(self.graph, source=coin1[1], target=coin2[1]))
                
                for path in paths:
                    # Calculate forward path weight
                    forward_weight = self.calculate_path_weight(path)
                    
                    # Calculate reverse path weight
                    reverse_path = path[::-1]
                    reverse_weight = self.calculate_path_weight(reverse_path)
                    
                    if forward_weight and reverse_weight:
                        total_factor = forward_weight * reverse_weight
                        if total_factor > min_profit_threshold:
                            opportunities.append({
                                'path': path,
                                'reverse_path': reverse_path,
                                'profit_factor': total_factor
                            })
            except Exception as e:
                continue
                
        return opportunities

    def calculate_path_weight(self, path):
        """Calculate the total weight of a path"""
        total_weight = 1
        try:
            for i in range(len(path) - 1):
                weight = self.graph[path[i]][path[i + 1]]['weight']
                total_weight *= weight
            return total_weight
        except:
            return None

    def execute_trade(self, path):
        """Execute trades for a given arbitrage path"""
        USD = "USD"
        first_buy = True
        
        for i in range(len(path)):
            current_ticker = path[i].upper()
            
            try:
                if first_buy:
                    # First trade is against USD
                    symbol = f"{current_ticker}{USD}"
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=1,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                    first_buy = False
                else:
                    # Subsequent trades are crypto-to-crypto
                    prev_ticker = path[i-1].upper()
                    symbol = f"{current_ticker}/{prev_ticker}"
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=1,
                        side='buy',
                        type='market',
                        time_in_force='gtc'
                    )
                
                st.success(f"Successfully ordered {current_ticker}")
                time.sleep(4)  # Rate limiting
                
            except Exception as e:
                st.error(f"Failed to order {current_ticker}: {str(e)}")
                break

def visualize_graph(graph):
    """Create an interactive visualization of the exchange rate graph"""
    pos = nx.spring_layout(graph)
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in graph.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(edge[2]['weight'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines')
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.upper())
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            size=20,
            color='#1f77b4',
            line_width=2))
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def main():
    st.title("Crypto Arbitrage Trading Bot")
    
    # Initialize app
    app = CryptoArbitrageApp()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    api_key = st.sidebar.text_input("Alpaca API Key", value="PKW96648RTR6XGCS65E2", type="password")
    api_secret = st.sidebar.text_input("Alpaca API Secret", value="Dj4GeLXnfSsdXzdPxV7UCE2N6WQTUUJEStKdzAaR", type="password")
    base_url = st.sidebar.text_input("Alpaca Base URL", value="https://paper-api.alpaca.markets")
    
    min_profit = st.sidebar.slider("Minimum Profit Threshold", 1.001, 1.05, 1.001, 0.001)
    
    if st.sidebar.button("Initialize Alpaca"):
        try:
            account = app.initialize_alpaca(api_key, api_secret, base_url)
            st.sidebar.success("Successfully connected to Alpaca!")
            st.sidebar.write(f"Account Status: {account.status}")
            st.sidebar.write(f"Buying Power: ${float(account.buying_power):.2f}")
        except Exception as e:
            st.sidebar.error(f"Failed to connect to Alpaca: {str(e)}")
    
    # Main app functionality
    if st.button("Fetch Latest Exchange Rates"):
        with st.spinner("Fetching exchange rates..."):
            exchange_rates = app.fetch_exchange_rates()
            app.update_graph(exchange_rates)
            
            # Visualize the graph
            st.subheader("Exchange Rate Graph")
            fig = visualize_graph(app.graph)
            st.plotly_chart(fig)
            
            # Find arbitrage opportunities
            opportunities = app.find_arbitrage_opportunities(min_profit_threshold=min_profit)
            
            if opportunities:
                st.subheader("Arbitrage Opportunities")
                for i, opp in enumerate(opportunities, 1):
                    with st.expander(f"Opportunity {i} - Profit Factor: {opp['profit_factor']:.4f}"):
                        st.write(f"Forward Path: {' → '.join(coin.upper() for coin in opp['path'])}")
                        st.write(f"Reverse Path: {' → '.join(coin.upper() for coin in opp['reverse_path'])}")
                        
                        if st.button(f"Execute Trade {i}"):
                            app.execute_trade(opp['path'])
            else:
                st.info("No arbitrage opportunities found above the profit threshold.")

if __name__ == "__main__":
    main()
