"""
ULTIMATE FLEXIBLE MASTER AGENT
Cascading Fallback System:
1. Pattern-based (FREE, instant)
2. Gemini API (uses quota, high quality)
3. Ollama Local AI (unlimited, no quota)
4. Helpful fallback message

Maximum flexibility - adapts to any situation!
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import json
import re
import warnings

warnings.filterwarnings('ignore')

# Load Gemini API keys
API_KEYS = []
for i in range(1, 20):
    if i == 1:
        key = os.getenv("GEMINI_API_KEY")
    else:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

# Gemini models
GEMINI_MODELS = ['gemini-2.5-flash', 'gemini-2.5-pro']

# Try imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class MasterAgent:
    """
    Ultimate Flexible Master Agent
    Cascading Fallback: Pattern → Gemini → Ollama → Message
    """
    
    def __init__(self):
        print("\n" + "=" * 80)
        print("🏥 PHARMACY AI - ULTIMATE FLEXIBLE SYSTEM")
        print("=" * 80)
        
        self.conversation_history = []
        
        # Load data
        print("\n📊 Loading data...")
        try:
            self.sales_df = pd.read_csv('data/sales_history.csv')
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
            self.inventory_df = pd.read_csv('data/current_inventory.csv')
            self.inventory_df['expiry_date'] = pd.to_datetime(self.inventory_df['expiry_date'])
            
            print(f"✅ Data loaded: {len(self.sales_df):,} sales, {len(self.inventory_df):,} inventory")
            self.data_available = True
        except Exception as e:
            print(f"⚠️ Data error: {e}")
            self.sales_df = pd.DataFrame()
            self.inventory_df = pd.DataFrame()
            self.data_available = False
        
        # Test Gemini API
        print(f"\n🌐 Testing Gemini API ({len(API_KEYS)} keys)...")
        self.gemini_combinations = []
        self.gemini_available = False
        
        if API_KEYS and GEMINI_AVAILABLE:
            self.genai = genai
            self.types = types
            
            # Quick test (only first 2 keys to save quota)
            for key in API_KEYS[:2]:
                for model in GEMINI_MODELS:
                    try:
                        client = genai.Client(api_key=key)
                        response = client.models.generate_content(
                            model=model,
                            contents='Test',
                            config=types.GenerateContentConfig(max_output_tokens=5, temperature=0)
                        )
                        if response and response.text:
                            self.gemini_combinations.append((key, model))
                            self.gemini_available = True
                            print(f"✅ Gemini API: {model} available")
                            break
                    except:
                        pass
                if self.gemini_combinations:
                    break
            
            if not self.gemini_available:
                print("⚠️ Gemini API: Quota exhausted or unavailable")
        else:
            if not API_KEYS:
                print("⚠️ Gemini API: No keys found")
            else:
                print("⚠️ Gemini API: Library not installed")
        
        # Test Ollama Local AI
        print("\n🤖 Testing Ollama Local AI...")
        self.ollama_available = False
        self.ollama_models = []
        
        if OLLAMA_AVAILABLE:
            try:
                models = ollama.list()
                if models and 'models' in models:
                    available = [m.get('model', m.get('name', '')) for m in models['models']]
                    
                    # Rankings: [speed_score, quality_score]
                    rankings = {
                        'qwen2.5:0.5b': (10, 6),
                        'tinyllama': (10, 5),
                        'gemma2:2b': (9, 7),
                        'llama3.2:1b': (9, 6),
                        'phi3:mini': (9, 8),
                        'phi3': (7, 8),
                        'llama3.2': (6, 9),
                        'mistral': (4, 10),
                    }
                    
                    for pattern, (speed, quality) in rankings.items():
                        matching = [m for m in available if pattern in m.lower()]
                        if matching:
                            self.ollama_models.append({
                                'name': matching[0],
                                'speed': speed,
                                'quality': quality,
                                'balanced_score': (speed + quality) / 2
                            })
                    
                    if self.ollama_models:
                        # Test first model
                        try:
                            response = ollama.generate(model=self.ollama_models[0]['name'], prompt='Test', stream=False)
                            if response:
                                self.ollama_available = True
                                print(f"✅ Ollama: {len(self.ollama_models)} models available")
                                for m in self.ollama_models[:3]:  # Show top 3
                                    print(f"   {m['name']}: Speed={m['speed']}/10, Quality={m['quality']}/10")
                        except:
                            pass
                    
                    if not self.ollama_available:
                        print("⚠️ Ollama: Models found but none working")
            except Exception as e:
                print(f"⚠️ Ollama: Not running - {str(e)[:50]}")
        else:
            print("⚠️ Ollama: Library not installed")
        
        # Mock agents
        self.available_agents = {
            'demand': self, 'transfer': self, 'supplier': self, 'capital': self,
            'inventory': self, 'pricing': self, 'prescription': self,
            'promotion': self, 'compliance': self, 'customer': self,
        }
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ SYSTEM READY: CASCADING FALLBACK MODE")
        print("\n📋 Available AI Methods (in priority order):")
        print("   1. Pattern-based: ✅ Always available (instant, FREE)")
        print(f"   2. Gemini API: {'✅ Available' if self.gemini_available else '❌ Not available'} {'(uses quota)' if self.gemini_available else ''}")
        print(f"   3. Ollama Local: {'✅ Available' if self.ollama_available else '❌ Not available'} {'(unlimited, FREE)' if self.ollama_available else ''}")
        print("   4. Fallback Message: ✅ Always available")
        print("=" * 80 + "\n")
    
    def ask(self, question):
        """Main ask method"""
        result = self.ask_concise(question)
        return result.get('answer', 'No response')
    
    def ask_concise(self, question):
        """Smart data-first with GenAI fallback"""
        result = {'answer': '', 'table_data': None, 'metrics': {}, 'agents_consulted': [], 'model_used': 'None'}
        
        if not self.data_available:
            result['answer'] = "Data unavailable"
            return result
        
        q = question.lower()
        
        # Check for migraine/headache/pain/fever medicines
        if any(w in q for w in ['migraine', 'headache', 'fever', 'pain']):
            items = self.inventory_df[self.inventory_df['product_name'].str.contains('paracetamol|ibuprofen|aspirin|sumatriptan|naproxen', case=False, na=False)]
            
            if not items.empty:
                # Have stock - show it
                ans = "**Medicines Available in Stock:**\n\n"
                for _, r in items.head(10).iterrows():
                    ans += f"- {r['product_name']}: ${r['unit_cost']:.2f}, Stock: {int(r['current_stock'])} units\n"
                return {'answer': ans, 'table_data': {'Product': items['product_name'].tolist(), 'Price': items['unit_cost'].tolist(), 'Stock': items['current_stock'].tolist()}, 'metrics': {}, 'agents_consulted': ['inventory'], 'model_used': 'Data'}
            else:
                # No stock - use GenAI for medical knowledge
                medical_prompt = f"List common medications for {question}. Provide medicine names and brief uses. Do not mention our inventory."
                
                # Try Gemini
                if self.gemini_available:
                    try:
                        gr = self._gemini_analysis(medical_prompt)
                        if gr:
                            gr['model_used'] = 'Gemini'
                            gr['answer'] = "**Not in current inventory. Common medications:**\n\n" + gr['answer']
                            return gr
                    except:
                        pass
                
                # Try Ollama
                if self.ollama_available and self.ollama_models:
                    for m in self.ollama_models:
                        try:
                            r = ""
                            for c in ollama.generate(model=m['name'], prompt=medical_prompt, stream=True, options={'num_predict': 400}):
                                if 'response' in c:
                                    r += c['response']
                            if r and len(r) > 30:
                                return {'answer': "**Not in current inventory. Common medications:**\n\n" + r.strip(), 'table_data': None, 'metrics': {}, 'agents_consulted': ['AI'], 'model_used': m['name'].split(':')[0]}
                        except:
                            continue
                
                return {'answer': "No stock available for these medications.", 'table_data': None, 'metrics': {}, 'agents_consulted': ['inventory'], 'model_used': 'Data'}
        
        # Ordering questions
        if 'should i order' in q or 'should we order' in q:
            result = self._handle_ordering(question)
            if result:
                result['model_used'] = 'Data'
                return result
        
        # Default - use GenAI
        if self.gemini_available:
            try:
                gr = self._gemini_analysis(question)
                if gr:
                    gr['model_used'] = 'Gemini'
                    return gr
            except:
                pass
        
        if self.ollama_available and self.ollama_models:
            for m in self.ollama_models:
                try:
                    r = ""
                    for c in ollama.generate(model=m['name'], prompt=question, stream=True, options={'num_predict': 500}):
                        if 'response' in c:
                            r += c['response']
                    if r and len(r) > 20:
                        return {'answer': r.strip(), 'table_data': None, 'metrics': {}, 'agents_consulted': ['AI'], 'model_used': m['name'].split(':')[0]}
                except:
                    continue
        
        result['answer'] = "Unable to process question"
        return result
    
    def ask(self, question):
        """Main ask method"""
        result = self.ask_concise(question)
        return result.get('answer', 'No response')
    
    def ask_concise(self, question):
        """Data-first system"""
        result = {'answer': '', 'table_data': None, 'metrics': {}, 'agents_consulted': [], 'model_used': 'None'}
        
        if not self.data_available:
            result['answer'] = "Data unavailable"
            return result
        
        q = question.lower()
        
        # Fever medications from inventory
        if 'fever' in q or 'pain' in q:
            items = self.inventory_df[self.inventory_df['product_name'].str.contains('paracetamol|ibuprofen|aspirin', case=False, na=False)]
            if not items.empty:
                ans = "**Medicines Available:**\n\n"
                for _, r in items.head(5).iterrows():
                    ans += f"- {r['product_name']}: ${r['unit_cost']:.2f}, Stock: {int(r['current_stock'])}\n"
                return {'answer': ans, 'table_data': {'Product': items['product_name'].tolist(), 'Price': items['unit_cost'].tolist()}, 'metrics': {}, 'agents_consulted': ['data'], 'model_used': 'Data'}
        
        # Try GenAI fallback
        if self.gemini_available:
            try:
                gr = self._gemini_analysis(question)
                if gr:
                    gr['model_used'] = 'Gemini'
                    return gr
            except:
                pass
        
        if self.ollama_available and self.ollama_models:
            for m in self.ollama_models:
                try:
                    r = ""
                    for c in ollama.generate(model=m['name'], prompt=question, stream=True):
                        if 'response' in c:
                            r += c['response']
                    if r and len(r) > 20:
                        return {'answer': r, 'table_data': None, 'metrics': {}, 'agents_consulted': ['AI'], 'model_used': m['name'].split(':')[0]}
                except:
                    continue
        
        return result
    
"""
ULTIMATE FLEXIBLE MASTER AGENT
Cascading Fallback System:
1. Pattern-based (FREE, instant)
2. Gemini API (uses quota, high quality)
3. Ollama Local AI (unlimited, no quota)
4. Helpful fallback message

Maximum flexibility - adapts to any situation!
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import json
import re
import warnings

warnings.filterwarnings('ignore')

# Load Gemini API keys
API_KEYS = []
for i in range(1, 20):
    if i == 1:
        key = os.getenv("GEMINI_API_KEY")
    else:
        key = os.getenv(f"GEMINI_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

# Gemini models
GEMINI_MODELS = ['gemini-2.5-flash', 'gemini-2.5-pro']

# Try imports
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class MasterAgent:
    """
    Ultimate Flexible Master Agent
    Cascading Fallback: Pattern → Gemini → Ollama → Message
    """
    
    def __init__(self):
        print("\n" + "=" * 80)
        print("🏥 PHARMACY AI - ULTIMATE FLEXIBLE SYSTEM")
        print("=" * 80)
        
        self.conversation_history = []
        
        # Load data
        print("\n📊 Loading data...")
        try:
            self.sales_df = pd.read_csv('data/sales_history.csv')
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
            self.inventory_df = pd.read_csv('data/current_inventory.csv')
            self.inventory_df['expiry_date'] = pd.to_datetime(self.inventory_df['expiry_date'])
            
            print(f"✅ Data loaded: {len(self.sales_df):,} sales, {len(self.inventory_df):,} inventory")
            self.data_available = True
        except Exception as e:
            print(f"⚠️ Data error: {e}")
            self.sales_df = pd.DataFrame()
            self.inventory_df = pd.DataFrame()
            self.data_available = False
        
        # Test Gemini API
        print(f"\n🌐 Testing Gemini API ({len(API_KEYS)} keys)...")
        self.gemini_combinations = []
        self.gemini_available = False
        
        if API_KEYS and GEMINI_AVAILABLE:
            self.genai = genai
            self.types = types
            
            # Quick test (only first 2 keys to save quota)
            for key in API_KEYS[:2]:
                for model in GEMINI_MODELS:
                    try:
                        client = genai.Client(api_key=key)
                        response = client.models.generate_content(
                            model=model,
                            contents='Test',
                            config=types.GenerateContentConfig(max_output_tokens=5, temperature=0)
                        )
                        if response and response.text:
                            self.gemini_combinations.append((key, model))
                            self.gemini_available = True
                            print(f"✅ Gemini API: {model} available")
                            break
                    except:
                        pass
                if self.gemini_combinations:
                    break
            
            if not self.gemini_available:
                print("⚠️ Gemini API: Quota exhausted or unavailable")
        else:
            if not API_KEYS:
                print("⚠️ Gemini API: No keys found")
            else:
                print("⚠️ Gemini API: Library not installed")
        
        # Test Ollama Local AI
        print("\n🤖 Testing Ollama Local AI...")
        self.ollama_available = False
        self.ollama_models = []
        
        if OLLAMA_AVAILABLE:
            try:
                models = ollama.list()
                if models and 'models' in models:
                    available = [m.get('model', m.get('name', '')) for m in models['models']]
                    
                    # Rankings: [speed_score, quality_score]
                    rankings = {
                        'qwen2.5:0.5b': (10, 6),
                        'tinyllama': (10, 5),
                        'gemma2:2b': (9, 7),
                        'llama3.2:1b': (9, 6),
                        'phi3:mini': (9, 8),
                        'phi3': (7, 8),
                        'llama3.2': (6, 9),
                        'mistral': (4, 10),
                    }
                    
                    for pattern, (speed, quality) in rankings.items():
                        matching = [m for m in available if pattern in m.lower()]
                        if matching:
                            self.ollama_models.append({
                                'name': matching[0],
                                'speed': speed,
                                'quality': quality,
                                'balanced_score': (speed + quality) / 2
                            })
                    
                    if self.ollama_models:
                        # Test first model
                        try:
                            response = ollama.generate(model=self.ollama_models[0]['name'], prompt='Test', stream=False)
                            if response:
                                self.ollama_available = True
                                print(f"✅ Ollama: {len(self.ollama_models)} models available")
                                for m in self.ollama_models[:3]:  # Show top 3
                                    print(f"   {m['name']}: Speed={m['speed']}/10, Quality={m['quality']}/10")
                        except:
                            pass
                    
                    if not self.ollama_available:
                        print("⚠️ Ollama: Models found but none working")
            except Exception as e:
                print(f"⚠️ Ollama: Not running - {str(e)[:50]}")
        else:
            print("⚠️ Ollama: Library not installed")
        
        # Mock agents
        self.available_agents = {
            'demand': self, 'transfer': self, 'supplier': self, 'capital': self,
            'inventory': self, 'pricing': self, 'prescription': self,
            'promotion': self, 'compliance': self, 'customer': self,
        }
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ SYSTEM READY: CASCADING FALLBACK MODE")
        print("\n📋 Available AI Methods (in priority order):")
        print("   1. Pattern-based: ✅ Always available (instant, FREE)")
        print(f"   2. Gemini API: {'✅ Available' if self.gemini_available else '❌ Not available'} {'(uses quota)' if self.gemini_available else ''}")
        print(f"   3. Ollama Local: {'✅ Available' if self.ollama_available else '❌ Not available'} {'(unlimited, FREE)' if self.ollama_available else ''}")
        print("   4. Fallback Message: ✅ Always available")
        print("=" * 80 + "\n")
    
    def ask(self, question):
        """Main ask method"""
        result = self.ask_concise(question)
        return result.get('answer', 'No response')
    
    def ask_concise(self, question):
        """
        CASCADING FALLBACK SYSTEM:
        1. Try Pattern-based (FREE, instant)
        2. Try Gemini API (uses quota, high quality)
        3. Try Ollama Local (unlimited, free)
        4. Helpful fallback message
        """
        
        result = {
            'answer': '',
            'table_data': None,
            'metrics': {},
            'agents_consulted': []
        }
        
        if not self.data_available:
            result['answer'] = "⚠️ Data not available."
            return result
        
        try:
            # ============ LEVEL 1: PATTERN-BASED (FREE, INSTANT) ============
            pattern_result = self._pattern_based_analysis(question)
            
            if not self._is_fallback_response(pattern_result['answer']):
                # Pattern worked! No need for AI
                pattern_result['agents_consulted'].append('(pattern-based)')
                return pattern_result
            
            # ============ LEVEL 2: GEMINI API (QUOTA, HIGH QUALITY) ============
            if self.gemini_available:
                gemini_result = self._gemini_analysis(question)
                
                if gemini_result and not self._is_fallback_response(gemini_result['answer']):
                    # Gemini worked!
                    gemini_result['agents_consulted'].append('(Gemini-API)')
                    return gemini_result
                else:
                    # Gemini failed (probably quota exhausted)
                    print("⚠️ Gemini quota exhausted, trying Ollama...")
            
            # ============ LEVEL 3: OLLAMA LOCAL (UNLIMITED, FREE) ============
            if self.ollama_available:
                ollama_result = self._ollama_analysis(question)
                
                if ollama_result and not self._is_fallback_response(ollama_result['answer']):
                    # Ollama worked!
                    ollama_result['agents_consulted'].append('(Ollama-local)')
                    return ollama_result
            
            # ============ LEVEL 4: HELPFUL FALLBACK ============
            result['answer'] = self._generate_helpful_fallback(question)
            result['agents_consulted'] = ['system']
            
        except Exception as e:
            result['answer'] = f"Error: {str(e)}"
            result['agents_consulted'] = ['system']
        
        return result
    
    def _is_fallback_response(self, answer):
        """Check if response is fallback"""
        if not answer:
            return True
        fallback_indicators = ['available analysis types', 'please rephrase', 'pattern_not_matched']
        return any(indicator in answer.lower() for indicator in fallback_indicators)
    
    def _gemini_analysis(self, question):
        """Analyze using Gemini API (Level 2)"""
        
        data_context = self._gather_data(question)
        
        prompt = f"""You are an expert pharmacy business intelligence consultant. Answer this question thoroughly and professionally.

QUESTION: {question}

PHARMACY DATA:
{json.dumps(data_context, indent=2, default=str)}

Provide a comprehensive, detailed answer as you would in a professional consulting context. Use specific product names, numbers, and insights from the data. Be thorough, clear, and actionable."""

        # Try each Gemini combination
        for key, model in self.gemini_combinations:
            try:
                client = self.genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        max_output_tokens=2000,
                        temperature=0.7
                    )
                )
                
                if response and response.text:
                    text = response.text.strip()
                    if text:
                        return {
                            'answer': text,
                            'table_data': self._generate_table(question),
                            'metrics': self._extract_metrics(text),
                            'agents_consulted': self._identify_agents(question)
                        }
            except Exception as e:
                # Quota exhausted or error, try next
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    print(f"   Gemini quota exhausted on this key, trying next...")
                continue
        
        # All Gemini attempts failed
        return None
    
    def _gather_data_for_ai(self, question):
        """Gather MINIMAL essential data for AI (faster)"""
        q = question.lower()
        summary = {}
        
        # Only send what's needed for this specific question
        
        # For product-specific questions
        product = self._extract_product(question)
        if product and not self.sales_df.empty:
            prod_sales = self.sales_df[self.sales_df['product_name'].str.contains(product, case=False, na=False)]
            if not prod_sales.empty:
                summary[product] = {
                    'avg_daily': float(prod_sales['quantity_sold'].mean()),
                    'stock': int(self.inventory_df[self.inventory_df['product_name'].str.contains(product, case=False, na=False)]['current_stock'].sum()) if not self.inventory_df.empty else 0
                }
                return summary  # Return early - don't gather more
        
        # For list/available questions - send top 10 only
        if any(w in q for w in ['available', 'names', 'list', 'which']):
            if 'painkiller' in q or 'pain' in q or 'fever' in q:
                keywords = ['paracetamol', 'ibuprofen', 'aspirin']
            elif 'antibiotic' in q:
                keywords = ['amoxicillin', 'azithromycin']
            elif 'vitamin' in q:
                keywords = ['vitamin']
            else:
                keywords = None
            
            if keywords and not self.inventory_df.empty:
                pattern = '|'.join(keywords)
                items = self.inventory_df[self.inventory_df['product_name'].str.contains(pattern, case=False, na=False)]
                if not items.empty:
                    summary['products'] = [
                        {'name': row['product_name'], 'stock': int(row['current_stock'])}
                        for _, row in items.head(10).iterrows()  # Top 10 only
                    ]
                    return summary
        
        # Generic context (minimal)
        if not self.sales_df.empty:
            summary['total_revenue'] = float(self.sales_df['total_amount'].sum())
        if not self.inventory_df.empty:
            summary['total_items'] = len(self.inventory_df)
        
        return summary
    
    def _select_best_model(self, question):
        """Select optimal model based on question complexity"""
        if not self.ollama_models:
            return None
        
        q = question.lower()
        
        # Complex indicators
        complex = ['compare', 'analyze', 'explain why', 'recommend', 'strategy', 'should i']
        # Simple indicators  
        simple = ['list', 'show', 'what are', 'names', 'which']
        
        is_complex = any(ind in q for ind in complex)
        is_simple = any(ind in q for ind in simple) and not is_complex
        
        if is_complex:
            # Sort by quality
            best = max(self.ollama_models, key=lambda x: x['quality'])
        elif is_simple:
            # Sort by speed
            best = max(self.ollama_models, key=lambda x: x['speed'])
        else:
            # Balanced
            best = max(self.ollama_models, key=lambda x: x['balanced_score'])
        
        return best['name']
    
    def _ollama_analysis(self, question):
        """Analyze using Ollama - intelligent model selection"""
        
        # Select best model for this question
        selected_model = self._select_best_model(question)
        if not selected_model:
            return None
        
        data_context = self._gather_data_for_ai(question)
        
        prompt = f"""As a pharmacy analyst, answer this question using the data provided. Be specific with product names and numbers, but stay concise.

Question: {question}

Data: {json.dumps(data_context, default=str)}"""

        try:
            full_response = ""
            stream = ollama.generate(
                model=selected_model,
                prompt=prompt,
                stream=True,
                options={
                    'temperature': 0.4,
                    'num_predict': 500,
                    'num_ctx': 2048,
                    'top_p': 0.9,
                    'repeat_penalty': 1.1,
                    'num_thread': 8
                }
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    full_response += chunk['response']
            
            answer = full_response.strip()
            
            if answer:
                return {
                    'answer': answer,
                    'table_data': self._generate_table(question),
                    'metrics': self._extract_metrics(answer),
                    'agents_consulted': self._identify_agents(question) + [f'(Ollama-{selected_model.split(":")[0]})']
                }
        except Exception as e:
            print(f"   Ollama error: {str(e)[:50]}")
            return None
        
        return None
    
    def _pattern_based_analysis(self, question):
        """Pattern-based analysis (Level 1)"""
        
        q = question.lower()
        
        if any(w in q for w in ['order', 'purchase']) and 'should' in q:
            return self._handle_ordering(question)
        elif any(w in q for w in ['demand', 'forecast']):
            return self._handle_demand(question)
        elif 'expir' in q:
            return self._handle_expiry(question)
        elif 'dead' in q and 'stock' in q:
            return self._handle_dead_stock()
        elif 'overstock' in q:
            return self._handle_overstock()
        elif 'dio' in q or 'capital' in q:
            return self._handle_capital()
        elif 'supplier' in q:
            return self._handle_supplier(question)
        elif 'transfer' in q:
            return self._handle_transfers()
        elif 'discount' in q or 'pric' in q:
            return self._handle_pricing()
        elif 'roi' in q or 'promotion' in q:
            return self._handle_promotion()
        else:
            return {'answer': 'PATTERN_NOT_MATCHED', 'table_data': None, 'metrics': {}, 'agents_consulted': []}
    
    def _generate_helpful_fallback(self, question):
        """Level 4: Helpful fallback when all else fails"""
        
        gemini_status = "Available but quota exhausted" if self.gemini_available else "Not configured"
        ollama_status = "Not available (install from ollama.com)" if not self.ollama_available else "Error occurred"
        
        return f"""**Question Received:** "{question}"

**Analysis Status:**
• Pattern-based: Question type not recognized
• Gemini API: {gemini_status}
• Ollama Local: {ollama_status}

**Available Question Types:**

**Inventory & Orders:**
• "Should I order X units of Y?"
• "What will be demand for X?"
• "What items expire in next 30 days?"
• "Show dead stock items"
• "Which medicines are overstocked?"

**Financial & Strategy:**
• "What is my DIO?"
• "Which supplier should I use?"
• "Which products should I discount?"
• "What is ROI of promotion?"
• "Recommend inter-store transfers"

**💡 Solutions:**
1. Rephrase question to match above types
2. Wait for Gemini quota reset (midnight UTC)
3. Install Ollama for unlimited local AI:
   - Download: https://ollama.com
   - Install model: `ollama pull llama3.2`
   - Restart this app"""
    
    # ==================== PATTERN HANDLERS ====================
    
    def _handle_ordering(self, question):
        """Ordering analysis"""
        product = self._extract_product(question)
        quantity = self._extract_quantity(question)
        
        if not product:
            return {'answer': 'PATTERN_NOT_MATCHED', 'table_data': None, 'metrics': {}, 'agents_consulted': []}
        
        prod_sales = self.sales_df[self.sales_df['product_name'].str.contains(product, case=False, na=False)]
        prod_inv = self.inventory_df[self.inventory_df['product_name'].str.contains(product, case=False, na=False)]
        
        if prod_sales.empty or prod_inv.empty:
            return {'answer': f"No data for {product}", 'agents_consulted': ['demand']}
        
        avg_daily = prod_sales['quantity_sold'].mean()
        current_stock = prod_inv['current_stock'].sum()
        days_supply = current_stock / avg_daily if avg_daily > 0 else 999
        
        if days_supply < 14:
            decision = "✅ YES - ORDER RECOMMENDED"
        elif days_supply > 60:
            decision = "⚠️ RECONSIDER"
        else:
            decision = "✅ YES - SAFE"
        
        answer = f"""**Order Analysis: {product}**

{decision}

• Current: {current_stock:.0f} units ({days_supply:.0f} days)
• Daily Sales: {avg_daily:.1f} units
• Order: {quantity} units

**Recommendation:** {'Order urgently' if days_supply < 14 else 'Reduce quantity' if days_supply > 60 else 'Proceed'}"""

        return {
            'answer': answer,
            'metrics': {'Stock': f"{current_stock:.0f}", 'Days': f"{days_supply:.0f}"},
            'agents_consulted': ['demand', 'inventory']
        }
    
    def _handle_demand(self, question):
        """Demand forecast"""
        product = self._extract_product(question)
        
        if not product:
            return {'answer': 'PATTERN_NOT_MATCHED', 'table_data': None, 'metrics': {}, 'agents_consulted': []}
        
        prod_sales = self.sales_df[self.sales_df['product_name'].str.contains(product, case=False, na=False)]
        
        if prod_sales.empty:
            return {'answer': f"No data for {product}", 'agents_consulted': ['demand']}
        
        avg_daily = prod_sales['quantity_sold'].mean()
        
        answer = f"""**Demand Forecast: {product}**

• Daily: {avg_daily:.1f} units
• 30-Day: {avg_daily * 30:.0f} units

**Recommendation:** Maintain {avg_daily * 14:.0f} units minimum"""

        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        
        return {
            'answer': answer,
            'table_data': {'Date': dates, 'Forecast': [round(avg_daily, 1)] * 30},
            'agents_consulted': ['demand']
        }
    
    def _handle_expiry(self, question):
        """Expiry analysis"""
        days = 60 if '60' in question else 90 if '90' in question else 30
        cutoff = datetime.now() + timedelta(days=days)
        expiring = self.inventory_df[self.inventory_df['expiry_date'] <= cutoff]
        
        if expiring.empty:
            return {'answer': f"✅ No items expiring in {days} days", 'agents_consulted': ['inventory']}
        
        value = (expiring['current_stock'] * expiring['unit_cost']).sum()
        
        answer = f"""**Expiring in {days} Days:**

• Items: {len(expiring)}
• Value at Risk: ${value:,.2f}

**Action:** Discount top items by value"""

        return {
            'answer': answer,
            'table_data': {
                'Product': expiring['product_name'].tolist()[:10],
                'Stock': expiring['current_stock'].tolist()[:10],
                'Expiry': expiring['expiry_date'].dt.strftime('%Y-%m-%d').tolist()[:10]
            },
            'agents_consulted': ['inventory']
        }
    
    def _handle_dead_stock(self):
        """Dead stock"""
        recent = self.sales_df[self.sales_df['date'] >= (datetime.now() - timedelta(days=90))]
        with_sales = set(recent['sku'].unique())
        dead = self.inventory_df[~self.inventory_df['sku'].isin(with_sales)]
        dead = dead[dead['current_stock'] > 0]
        
        if dead.empty:
            return {'answer': "✅ No dead stock", 'agents_consulted': ['inventory']}
        
        locked = (dead['current_stock'] * dead['unit_cost']).sum()
        
        return {
            'answer': f"**Dead Stock:**\n\n• Items: {len(dead)}\n• Locked: ${locked:,.2f}\n\n**Action:** Aggressive clearance",
            'table_data': {
                'Product': dead['product_name'].tolist()[:10],
                'Stock': dead['current_stock'].tolist()[:10]
            },
            'agents_consulted': ['inventory']
        }
    
    def _handle_overstock(self):
        """Overstock"""
        high = self.inventory_df.nlargest(10, 'current_stock')
        return {
            'answer': "**Highest Stock Items:**\n\nReview for overstock",
            'table_data': {'Product': high['product_name'].tolist(), 'Stock': high['current_stock'].tolist()},
            'agents_consulted': ['inventory']
        }
    
    def _handle_capital(self):
        """DIO"""
        total = (self.inventory_df['current_stock'] * self.inventory_df['unit_cost']).sum()
        daily_cogs = self.sales_df.groupby('date')['total_amount'].sum().mean() * 0.7
        dio = total / daily_cogs if daily_cogs > 0 else 40
        
        return {
            'answer': f"**DIO Analysis:**\n\n• Inventory: ${total:,.2f}\n• DIO: {dio:.0f} days\n• Target: 30-45 days",
            'metrics': {'DIO': f"{dio:.0f} days"},
            'agents_consulted': ['capital']
        }
    
    def _handle_supplier(self, question):
        """Supplier"""
        return {
            'answer': "**Supplier Selection:**\n\n• Reliability: 30%\n• Lead Time: 20%\n• Cost: 15%\n• Quality: 20%\n• Compliance: 15%\n\n**Strategy:** Primary (70%) + Backup (30%)",
            'agents_consulted': ['supplier']
        }
    
    def _handle_transfers(self):
        """Transfers"""
        return {
            'answer': "**Transfer Strategy:**\n\n1. Expiring items → High-demand store\n2. Slow movers → Faster store\n3. Balance monthly\n\n**Criteria:** Savings > Cost",
            'agents_consulted': ['transfer']
        }
    
    def _handle_pricing(self):
        """Pricing"""
        return {
            'answer': "**Pricing Strategy:**\n\n• Near Expiry: 20-40% off\n• Slow Movers: 15-25%\n• Fast Movers: 5-10%",
            'agents_consulted': ['pricing']
        }
    
    def _handle_promotion(self):
        """ROI"""
        return {
            'answer': "**ROI Framework:**\n\nROI = (Revenue - Cost) / Cost × 100%\n\n• Excellent: ≥200%\n• Good: 100-200%\n• Moderate: 50-100%",
            'agents_consulted': ['promotion']
        }
    
    # ==================== HELPERS ====================
    
    def _gather_data(self, question):
        """Gather data for AI"""
        summary = {}
        q = question.lower()
        
        # Overall metrics
        if not self.sales_df.empty:
            summary['sales'] = {
                'total_revenue': float(self.sales_df['total_amount'].sum()),
                'avg_daily': float(self.sales_df.groupby('date')['total_amount'].sum().mean())
            }
        
        if not self.inventory_df.empty:
            summary['inventory'] = {
                'items': len(self.inventory_df),
                'value': float((self.inventory_df['current_stock'] * self.inventory_df['unit_cost']).sum())
            }
        
        # Product-specific data
        product = self._extract_product(question)
        if product:
            prod_sales = self.sales_df[self.sales_df['product_name'].str.contains(product, case=False, na=False)]
            if not prod_sales.empty:
                summary[f'{product}'] = {'avg_daily_sales': float(prod_sales['quantity_sold'].mean())}
        
        # If asking about "available", "names", "list" - send actual product list
        if any(w in q for w in ['available', 'names', 'list', 'which', 'what medicines', 'what drugs']):
            if not self.inventory_df.empty:
                # Get top products by value
                inv_copy = self.inventory_df.copy()
                inv_copy['total_value'] = inv_copy['current_stock'] * inv_copy['unit_cost']
                top_products = inv_copy.nlargest(20, 'total_value')
                
                summary['available_products'] = [
                    {
                        'name': row['product_name'],
                        'stock': int(row['current_stock']),
                        'category': row.get('category', 'N/A')
                    }
                    for _, row in top_products.iterrows()
                ]
        
        # If asking about categories (painkillers, antibiotics, etc)
        if any(w in q for w in ['painkiller', 'antibiotic', 'vitamin', 'fever', 'cold', 'cough']):
            if not self.inventory_df.empty:
                # Filter by category or name
                if 'painkiller' in q or 'pain' in q:
                    mask = self.inventory_df['product_name'].str.contains('paracetamol|ibuprofen|aspirin|diclofenac', case=False, na=False)
                elif 'antibiotic' in q:
                    mask = self.inventory_df['product_name'].str.contains('amoxicillin|azithromycin|ciprofloxacin', case=False, na=False)
                elif 'vitamin' in q:
                    mask = self.inventory_df['product_name'].str.contains('vitamin|multivitamin', case=False, na=False)
                elif 'fever' in q:
                    mask = self.inventory_df['product_name'].str.contains('paracetamol|ibuprofen', case=False, na=False)
                else:
                    mask = None
                
                if mask is not None:
                    category_products = self.inventory_df[mask]
                    if not category_products.empty:
                        summary['category_specific_products'] = [
                            {
                                'name': row['product_name'],
                                'stock': int(row['current_stock']),
                                'unit_cost': float(row['unit_cost'])
                            }
                            for _, row in category_products.head(15).iterrows()
                        ]
        
        return summary
    
    def _generate_table(self, question):
        """Generate table"""
        if 'forecast' in question.lower() or 'demand' in question.lower():
            product = self._extract_product(question)
            if product:
                prod = self.sales_df[self.sales_df['product_name'].str.contains(product, case=False, na=False)]
                if not prod.empty:
                    avg = prod['quantity_sold'].mean()
                    dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
                    return {'Date': dates, 'Forecast': [round(avg, 1)] * 30}
        return None
    
    def _extract_metrics(self, text):
        """Extract metrics"""
        metrics = {}
        curr = re.findall(r'\$[\d,]+', text)
        if curr: metrics['Value'] = curr[0]
        return metrics
    
    def _identify_agents(self, question):
        """Identify agents"""
        q = question.lower()
        agents = []
        if any(w in q for w in ['demand', 'forecast']): agents.append('demand')
        if any(w in q for w in ['stock', 'inventory']): agents.append('inventory')
        return agents if agents else ['general']
    
    def _extract_product(self, text):
        """Extract product"""
        common = ['paracetamol', 'ibuprofen', 'aspirin', 'amoxicillin']
        for p in common:
            if p in text.lower():
                return p.capitalize()
        return None
    
    def _extract_quantity(self, text):
        """Extract quantity"""
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else 1000
    
    def get_conversation_history(self):
        return self.conversation_history
    
    def clear_history(self):
        self.conversation_history = []


if __name__ == "__main__":
    print("Testing Ultimate Flexible Master Agent...")
    try:
        master = MasterAgent()
        
        if master.data_available:
            questions = [
                "Should I order 1000 units of Paracetamol?",  # Pattern
                "What medicines for fever?",  # Gemini → Ollama
            ]
            
            for q in questions:
                print(f"\n{'='*70}")
                print(f"Q: {q}")
                print('='*70)
                result = master.ask_concise(q)
                print(result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer'])
                print(f"\nMethod used: {result['agents_consulted']}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
