AI文明がどのようにして人間を超える解法を発見しうるか、そのプロセスを具体的にシミュレートするものです。ローカルマシンで直接実行し、その挙動を観察できるように設計されています。

プロトタイプの共通思想
LLMの役割: LLMは直接問題を解くのではなく、問題に対する**「戦略」や「ヒューリスティック」、「代理モデル」**をコードとして生成・提案する役割を担います。ここではその創造的なプロセスをLLM_propose_strategy関数でシミュレートします。

進化のプロセス: 各AI文明（エージェント群）が提案した戦略を「るつぼ（Crucible）」環境で実行・評価します。より優れた成果を出した戦略が「進化的エンジン」によって選択され、次の世代の戦略の土台となります。

目的: これらのプロトタイプは、問題を完全に解くことではなく、より優れた解法発見プロセスを自動化・加速できるかを検証することを目的としています。

このプロトタイプでは、AI文明に、計算コストの高いN体シミュレーション（重力計算）を高速化するための代理モデル（Surrogate Model）を発明させます。フィットネスは、代理モデルの予測精度と計算速度のバランスで評価されます。

## セットアップ

### 必要条件
- Python 3.10以上
- Google AI API キー（無料）

### インストール手順

1. **リポジトリのクローン**
```bash
git clone https://github.com/TheIllusionOfLife/Galaxy.git
cd Galaxy
```

2. **仮想環境の作成と有効化**
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

3. **依存関係のインストール**
```bash
pip install -e .
```

4. **API キーの設定**
- [Google AI Studio](https://aistudio.google.com/apikey) で無料のAPI キーを取得
- プロジェクトルートに `.env` ファイルを作成:
```bash
cp .env.example .env
```
- `.env` ファイルを編集して、API キーを設定:
```
GOOGLE_API_KEY=your_api_key_here
```

## 使用方法

### 基本的な実行

```bash
# 仮想環境を有効化（まだの場合）
source .venv/bin/activate

# 進化的最適化を実行
python prototype.py
```

### 実行結果

プログラムは以下を出力します：
- 各世代の評価結果（フィットネス、精度、速度）
- トップパフォーマンスのモデル
- LLM使用統計（トークン数、コスト、成功率）

### コスト管理

- **無料枠**: 1日1,000リクエスト、毎分15リクエストまで
- **デフォルト設定**: 1実行あたり50 APIコール
- **実行コスト**: 約$0.02/実行（予算の2%）
- **レート制限**: 自動的に15 RPMを維持

### テスト

API接続をテスト:
```bash
python test_gemini_connection.py
```

単体テストを実行:
```bash
pytest tests/
```

## Session Handover

### Last Updated: October 27, 2025 10:04 PM JST

#### Recently Completed
- ✅ [PR #5]: CI/CD infrastructure with automated testing and code quality pipeline
  - Implemented GitHub Actions CI (Python 3.10, 3.11, 3.12 matrix testing)
  - Added Ruff (linting/formatting), Mypy (type checking), Pytest (testing)
  - Created pre-commit hooks, Makefile, and CONTRIBUTING.md
  - Fixed all type errors in critical modules (code_validator, gemini_client, config)
  - Translated all Japanese comments to English per PR review feedback
  - All CI checks passing across all Python versions
- ✅ [PR #2]: Gemini 2.5 Flash Lite API integration with comprehensive security validation
  - Implemented multi-layer code validation (AST-based + runtime sandbox)
  - Added rate limiting (15 RPM), cost tracking, and budget enforcement
  - Fixed TOCTOU vulnerability, rate limiter retry logic, SAFE_BUILTINS security issue

#### Next Priority Tasks
1. **[LLM Code Evolution]**: Run full evolutionary cycle with real Gemini API
   - Source: PR #5 merged, ready for production use with CI/CD
   - Context: Core infrastructure complete with automated testing
   - Approach: Execute `python prototype.py` and monitor LLM-generated surrogate models

2. **[Documentation]**: Add ARCHITECTURE.md explaining system design
   - Source: Previous session recommendation
   - Context: Code is production-ready but lacks architecture overview
   - Approach: Document: LLM → Validator → Sandbox → Evolution Engine flow

3. **[Test Coverage]**: Expand test coverage with more integration tests
   - Source: CI/CD infrastructure now in place
   - Context: Basic tests exist, but could expand coverage
   - Approach: Add more test scenarios, edge cases, error conditions

#### Known Issues / Blockers
- None currently - all critical issues resolved
- claude-review workflow has expected failure (workflow validation issue, non-blocking)

#### Session Learnings
- **CI/CD Infrastructure**: Comprehensive setup with Ruff + Mypy + Pytest provides strong foundation
- **Type Safety**: Strict Mypy on critical modules (code_validator, gemini_client, config) catches errors early
- **PR Review Systematic**: `/fix_pr_graphql` with 5-item verification checklist ensures no feedback missed
- **Japanese→English Translation**: All comments translated improves international collaboration
- **Modern Python Syntax**: Updated isinstance to use union syntax (X | Y) for Python 3.10+
