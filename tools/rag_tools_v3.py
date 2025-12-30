"""
RAG Tools V3 - Adapted for the Generic ReactAgent Framework
"""
from typing import Dict, Any, List
import yfinance as yf

# Using the new decorator from ReactAgent
from ReactAgent.decorators import tool
from embedding_manager.embedding_manager import EmbeddingManager


# =========================
# TOOLS
# =========================

# Global variable to store embedding_manager
_embedding_manager = None


def initialize_rag_tools(embedding_manager: EmbeddingManager):
    """
    Initialize RAG tools with the embedding manager
    
    Args:
        embedding_manager: EmbeddingManager instance
    """
    global _embedding_manager
    _embedding_manager = embedding_manager


@tool()
def search_documents(query: str) -> Dict[str, Any]:
    """
    Search relevant documents in the knowledge base using semantic search.
    
    Args:
        query: Query string to search for.
    """
    if _embedding_manager is None:
        return {
            "success": False,
            "error": "EmbeddingManager not initialized. Call initialize_rag_tools() first."
        }
    
    try:
        print(f"🔍 [DEBUG] search_documents called with: '{query}'")

        # Search in Qdrant
        results = _embedding_manager.search(query=query, top_k=3)
        
        # Format response
        chunks = [
            {
                "content": r["content"],
                "score": r["score"],
                "metadata": r["metadata"]
            }
            for r in results
        ]
        
        return {
            "success": True,
            "query": query,
            "results": chunks,
            "total_found": len(chunks)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "query": query
        }


@tool()
def get_stock_price(ticker: str, period: str = "1mo") -> Dict[str, Any]:
    """
    Get price information for a SINGLE stock.
    
    Args:
        ticker: Stock ticker (e.g: AAPL, GOOGL, MSFT, PETR4.SA).
        period: Historical period (1d, 5d, 1mo, 3mo, 6mo, 1y). Default is '1mo'.
    """
    print(f"🔍 [DEBUG] get_stock_price called with: ticker={ticker}, period={period}")
    try:
        ticker = ticker.upper()
        
        # Determine strict suffix for Brazilian stocks if seemingly missing
        # Simple heuristic: if 4 letters + number, try appending .SA if logic requires,
        # but yfinance often handles it. We will trust user input but maybe clean it.
        
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        
        if hist.empty:
            return {
                "success": False,
                "error": f"No data found for {ticker}",
                "message": f"Could not find price data for {ticker}. Please check if the ticker is correct."
            }
        
        info = stock.info
        last_close = float(hist['Close'].iloc[-1])
        
        # Calculate change
        if len(hist) > 1:
            first_close = float(hist['Close'].iloc[0])
            change = last_close - first_close
            change_percent = (change / first_close) * 100
        else:
            change = 0
            change_percent = 0
        
        result = {
            "success": True,
            "ticker": ticker,
            "company_name": info.get('longName', ticker),
            "current_price": round(last_close, 2),
            "currency": info.get('currency', 'USD'),
            "period": period,
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
            "summary": f"{info.get('longName', ticker)} is trading at {round(last_close, 2)} {info.get('currency', 'USD')}."
        }
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "ticker": ticker
        }


@tool()
def compare_stocks(tickers: List[str], period: str = "1mo") -> Dict[str, Any]:
    """
    Compare the performance of MULTIPLE stocks (2 or more).
    
    Args:
        tickers: List of stock symbols (e.g: ['AAPL', 'GOOGL', 'MSFT']).
        period: Comparison period (1d, 5d, 1mo, 3mo, 6mo, 1y). Default is '1mo'.
    """
    print(f"🔍 [DEBUG] compare_stocks called with: tickers={tickers}, period={period}")
    try:
        # Pydantic validation guarantees list of strings if properly hinted, but safe check:
        # In rag_tools.py logic there was flexible input handling. 
        # The new @tool relies on Pydantic validation of arguments.
        
        ticker_list = [t.strip().upper() for t in tickers]
        
        if len(ticker_list) < 2:
            return {
                "success": False,
                "error": "Please provide at least 2 tickers to compare."
            }
        
        results = []
        failed_tickers = []
        
        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    first_close = float(hist['Close'].iloc[0])
                    last_close = float(hist['Close'].iloc[-1])
                    change_percent = ((last_close - first_close) / first_close) * 100
                    
                    results.append({
                        "ticker": ticker,
                        "start_price": round(first_close, 2),
                        "end_price": round(last_close, 2),
                        "change_percent": round(change_percent, 2)
                    })
                else:
                    failed_tickers.append(ticker)
            except Exception:
                failed_tickers.append(ticker)
        
        if not results:
             return {
                "success": False,
                "error": "Could not find data for any of the requested stocks."
            }
            
        # Sort by performance
        results.sort(key=lambda x: x['change_percent'], reverse=True)
        
        best = results[0]
        worst = results[-1]
        
        summary = f"In the {period} period, {best['ticker']} performed best ({best['change_percent']}%), while {worst['ticker']} performed worst ({worst['change_percent']}%)."
        
        if failed_tickers:
            summary += f" Note: Could not fetch data for: {', '.join(failed_tickers)}."
            
        return {
            "success": True,
            "period": period,
            "stocks": results,
            "best_performer": best,
            "worst_performer": worst,
            "summary": summary
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool()
def redirect(reason: str = "out of scope") -> Dict[str, Any]:
    """
    Use when the question is NOT related to finance, economy, stock market, or investments.
    
    Args:
        reason: Reason why the topic is out of scope.
    """
    return {
        "success": True,
        "redirected": True,
        "reason": reason,
        "message": "Question is out of financial/economic scope."
    }
