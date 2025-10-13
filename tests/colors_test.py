#!/usr/bin/env python3
"""
Color testing utility for SABRE client themes.

Run this to see how different colors look in light vs dark mode.
Usage: uv run python tests/colors_test.py
"""

from sabre.client.tui import TUI


def main():
    """Main entry point - auto-detects terminal and shows appropriate theme"""
    # Create TUI instance (auto-detects theme)
    tui = TUI()
    term_info = tui.get_terminal_info()
    detected_theme = term_info["detected_theme"]

    # Show terminal detection info
    print("Terminal Detection:")
    print(f"  Terminal: {term_info['term_program']}")
    print(f"  TERM: {term_info['term']}")
    print(f"  Size: {term_info['size'].columns}x{term_info['size'].lines}")
    print(f"  COLORFGBG: {term_info['colorfgbg']}")
    print(f"  iTerm Profile: {term_info['iterm_profile']}")
    print(f"  Detected theme: {detected_theme}")
    print(f"  Detection method: {term_info['detection_method']}")
    print()
    print(f"═══ {detected_theme.upper()} MODE ═══")
    print()

    # Show color assignments
    print("Current Color Assignments:")
    for name, color in tui.colors.items():
        if name == "pygments_style":
            continue
        print(f"  {name}: {color}")
    print()

    # First exchange - simple question
    print("Example Conversation:")
    print()

    # User input
    user_color = tui.colors["user_input"]
    tui.print(f'<style fg="{user_color}">&gt; what is your name?</style>')
    print()

    # RESPONSE_ROUND
    path = ["71a036", "bd01c0"]
    tui.print_tree_node("RESPONSE_ROUND", "iteration 1", depth=1, path=path)
    tui.render_response_start({"model": "gpt-4o", "prompt_tokens": 0}, depth=1)

    # RESPONSE_TEXT
    tui.print_tree_node(
        "RESPONSE_TEXT", "104 chars, no helpers • Tokens: 9963 in, 26 out, 0 reasoning", depth=1, path=path
    )
    tui.render_response_text(
        {
            "text": "I'm an AI assistant and I don't have a personal name, but you can call me Assistant. How can I help you?"
        },
        depth=1,
    )

    # COMPLETE
    print()
    tui.print_tree_node("COMPLETE", "", depth=1, path=path)
    tui.render_complete(
        {
            "final_message": "I'm an AI assistant and I don't have a personal name, but you can call me Assistant. How can I help you?"
        },
        depth=1,
    )

    # Second exchange - graph plotting with helpers
    print()
    tui.print(f'<style fg="{user_color}">&gt; render a graph of x^5+y^5 = 1</style>')
    print()

    path2 = ["f8fb15", "c2342b"]

    # RESPONSE_ROUND
    tui.print_tree_node("RESPONSE_ROUND", "iteration 1", depth=1, path=path2)
    tui.render_response_start({"model": "gpt-4o", "prompt_tokens": 0}, depth=1)

    # RESPONSE_TEXT
    tui.print_tree_node(
        "RESPONSE_TEXT", "770 chars, 1 helper(s) • Tokens: 10009 in, 268 out, 0 reasoning", depth=1, path=path2
    )
    tui.render_response_text(
        {"text": "To render a graph of the equation \\(x^5 + y^5 = 1\\), I'll create a plot for you."}, depth=1
    )

    # HELPER_BLOCK
    helper_path = ["c2342b", "64ba20"]
    tui.print_tree_node("HELPER_BLOCK", "#1", depth=2, path=helper_path)

    # Code block
    code = """import numpy as np
import matplotlib.pyplot as plt

# Define the function for x^5 + y^5 = 1
def func(x):
    return (1 - x**5)**(1/5)

# Create a range of x values
x = np.linspace(-1, 1, 400)
y_positive = func(x)
y_negative = -func(x)

# Plotting the curve
with matplotlib_to_image(figsize=(6, 6), dpi=130):
    plt.plot(x, y_positive, label=r'$x^5 + y^5 = 1$', color='b')
    plt.plot(x, y_negative, color='b')
    plt.title('Graph of $x^5 + y^5 = 1$')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)"""

    tui.print_code_block(code, depth=2)

    # EXECUTING
    tui.print_tree_node("EXECUTING", "...", depth=2, path=helper_path)

    # RESULT
    tui.print_tree_node(
        "RESULT #1", "455ms - import numpy as np import matplotlib.pyplot as plt...", depth=2, path=helper_path
    )
    tui.render_helpers_end({"result": [{"type": "text", "data": "[Generated 1 matplotlib figure(s)]"}]}, depth=2)

    # Second RESPONSE_ROUND
    path3 = ["f8fb15", "cbc7a8"]
    tui.print_tree_node("RESPONSE_ROUND", "iteration 2", depth=1, path=path3)
    tui.render_response_start({"model": "gpt-4o", "prompt_tokens": 0}, depth=1)

    # RESPONSE_TEXT (iteration 2)
    tui.print_tree_node(
        "RESPONSE_TEXT", "242 chars, no helpers • Tokens: 10385 in, 97 out, 0 reasoning", depth=1, path=path3
    )
    tui.render_response_text(
        {
            "text": "Here is the graph of the equation \\(x^5 + y^5 = 1\\):\n![Graph](http://localhost:8011/files/conv_68e84aa1dabc8190b7c7c7912d5953a80e47f52915c448ba/figure_1_1.png)\nLet me know if there's anything else you'd like to explore!"
        },
        depth=1,
    )

    # COMPLETE
    print()
    tui.print_tree_node("COMPLETE", "", depth=1, path=path3)
    tui.render_complete(
        {
            "final_message": "Here is the graph of the equation \\(x^5 + y^5 = 1\\):\nLet me know if there's anything else you'd like to explore!"
        },
        depth=1,
    )

    # Show error and warning examples
    print()
    print("Error and Warning Examples:")
    print()

    # ERROR
    error_path = ["abc123", "def456"]
    tui.print_tree_node("ERROR", "", depth=1, path=error_path)
    tui.print_content_block(["NameError: name 'x' is not defined"], depth=1, color="error")

    # RATE_LIMIT
    tui.print_tree_node("RATE_LIMIT", "retrying in 2.0s (attempt 1/3)", depth=1, path=error_path)
    tui.render_response_retry({"reason": "Rate limit exceeded"}, depth=1)

    print()


if __name__ == "__main__":
    main()
