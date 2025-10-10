#!/usr/bin/env python3
"""
Color testing utility for SABRE client themes.

Run this to see how different colors look in light vs dark mode.
Usage: uv run python tests/colors_test.py
"""

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import HTML

# Import TUI and COLORS from sabre.client.tui
from sabre.client.tui import TUI, COLORS

# All available ANSI colors for reference
ALL_ANSI_COLORS = [
    # Standard colors
    "ansiblack",
    "ansired",
    "ansigreen",
    "ansiyellow",
    "ansiblue",
    "ansimagenta",
    "ansicyan",
    "ansiwhite",
    # Bright colors
    "ansibrightblack",
    "ansibrightred",
    "ansibrightgreen",
    "ansibrightyellow",
    "ansibrightblue",
    "ansibrightmagenta",
    "ansibrightcyan",
    "ansibrightwhite",
    # Default
    "ansidefault",
]


def show_theme_colors(theme: str):
    """Display all colors for a specific theme using actual TUI rendering methods"""
    # Create a TUI instance
    tui = TUI()
    # Override theme for testing
    tui.theme = theme
    tui.colors = COLORS[theme]

    print_formatted_text(HTML(f'\n<b><style fg="ansicyan">═══ {theme.upper()} MODE ═══</style></b>\n'))

    # Show current color assignments
    print_formatted_text(HTML("<b>Current Color Assignments:</b>"))
    for name, color in tui.colors.items():
        if name == "pygments_style":
            continue
        sample_text = f"{name}: This is sample text"
        print_formatted_text(HTML(f'  <style fg="{color}">{sample_text}</style> (color: {color})'))

    print_formatted_text(HTML("\n<b>Example Conversation:</b>\n"))

    # User input
    user_color = tui.colors["user_input"]
    tui.print(f'<style fg="{user_color}">&gt; what is your name?</style>\n')

    # First exchange - simple question
    path = ["71a036", "bd01c0"]

    # RESPONSE_ROUND
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
    tui.print()
    tui.print_tree_node("COMPLETE", "", depth=1, path=path)
    tui.render_complete(
        {
            "final_message": "I'm an AI assistant and I don't have a personal name, but you can call me Assistant. How can I help you?"
        },
        depth=1,
    )

    # Second exchange - graph plotting with helpers
    tui.print(f'\n<style fg="{user_color}">&gt; render a graph of x^5+y^5 = 1</style>\n')

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
    tui.print()
    tui.print_tree_node("COMPLETE", "", depth=1, path=path3)
    tui.render_complete(
        {
            "final_message": "Here is the graph of the equation \\(x^5 + y^5 = 1\\):\nLet me know if there's anything else you'd like to explore!"
        },
        depth=1,
    )

    # Show error and warning examples
    print_formatted_text(HTML("\n<b>Error and Warning Examples:</b>"))

    # ERROR
    error_path = ["abc123", "def456"]
    tui.print_tree_node("ERROR", "", depth=1, path=error_path)
    tui.print_content_block(["NameError: name 'x' is not defined"], depth=1, color="error")

    # RATE_LIMIT
    tui.print_tree_node("RATE_LIMIT", "retrying in 2.0s (attempt 1/3)", depth=1, path=error_path)
    tui.render_response_retry({"reason": "Rate limit exceeded"}, depth=1)


def show_all_ansi_colors():
    """Show all available ANSI colors"""
    print_formatted_text(HTML('\n<b><style fg="ansicyan">═══ ALL AVAILABLE ANSI COLORS ═══</style></b>\n'))

    print_formatted_text(HTML("<b>Standard Colors:</b>"))
    for color in ALL_ANSI_COLORS[:8]:
        print_formatted_text(HTML(f'  <style fg="{color}">████████</style> {color}'))

    print_formatted_text(HTML("\n<b>Bright Colors:</b>"))
    for color in ALL_ANSI_COLORS[8:16]:
        print_formatted_text(HTML(f'  <style fg="{color}">████████</style> {color}'))

    print_formatted_text(HTML("\n<b>Default:</b>"))
    print_formatted_text(HTML('  <style fg="ansidefault">████████</style> ansidefault'))

    # Show with sample text
    print_formatted_text(HTML("\n<b>Sample Text in Each Color:</b>"))
    sample = "The quick brown fox jumps over the lazy dog"
    for color in ALL_ANSI_COLORS:
        print_formatted_text(HTML(f'  <style fg="{color}">{sample}</style>'))
        print_formatted_text(HTML(f'  <style fg="ansibrightblack">  → {color}</style>'))


def show_color_combinations():
    """Show different color combinations for key UI elements"""
    print_formatted_text(HTML('\n<b><style fg="ansicyan">═══ COLOR COMBINATION IDEAS ═══</style></b>\n'))

    combinations = [
        {
            "name": "Current Dark",
            "node_label": "ansibrightcyan",
            "result": "ansibrightgreen",
            "complete": "ansiwhite",
        },
        {
            "name": "Muted Dark",
            "node_label": "ansicyan",
            "result": "ansigreen",
            "complete": "ansiwhite",
        },
        {
            "name": "Blue Dark",
            "node_label": "ansibrightblue",
            "result": "ansibrightcyan",
            "complete": "ansiwhite",
        },
        {
            "name": "Magenta Dark",
            "node_label": "ansibrightmagenta",
            "result": "ansibrightgreen",
            "complete": "ansiwhite",
        },
        {
            "name": "Yellow Dark",
            "node_label": "ansibrightyellow",
            "result": "ansibrightcyan",
            "complete": "ansiwhite",
        },
    ]

    for combo in combinations:
        print_formatted_text(HTML(f"\n<b>{combo['name']}:</b>"))
        node_color = combo["node_label"]
        result_color = combo["result"]
        complete_color = combo["complete"]

        print_formatted_text(HTML(f'⏺ <style fg="{node_color}">[RESPONSE_ROUND]</style> iteration 1'))
        print_formatted_text(HTML(f'  <style fg="{result_color}">⎿  Model: gpt-4o</style>'))
        print_formatted_text(HTML(f'⏺ <style fg="{node_color}">[COMPLETE]</style>'))
        print_formatted_text(HTML(f'  <style fg="{complete_color}">⎿  Here is your final answer.</style>'))
        print_formatted_text(
            HTML(
                f'  <style fg="ansibrightblack">  (node: {node_color}, result: {result_color}, complete: {complete_color})</style>'
            )
        )


def main():
    """Main entry point - auto-detects terminal and shows appropriate theme"""
    # Create TUI instance (auto-detects theme)
    tui = TUI()
    term_info = tui.get_terminal_info()
    detected_theme = term_info["detected_theme"]

    # Show terminal detection info
    print_formatted_text(HTML('<b><style fg="ansicyan">Terminal Detection</style></b>'))
    print_formatted_text(HTML(f'  <style fg="ansibrightblack">Terminal: {term_info["term_program"]}</style>'))
    print_formatted_text(HTML(f'  <style fg="ansibrightblack">TERM: {term_info["term"]}</style>'))
    print_formatted_text(
        HTML(f'  <style fg="ansibrightblack">Size: {term_info["size"].columns}x{term_info["size"].lines}</style>')
    )
    print_formatted_text(HTML(f'  <style fg="ansibrightblack">COLORFGBG: {term_info["colorfgbg"]}</style>'))
    print_formatted_text(HTML(f'  <style fg="ansibrightblack">iTerm Profile: {term_info["iterm_profile"]}</style>'))
    print_formatted_text(HTML(f'  <style fg="ansicyan">Detected theme: {detected_theme}</style>'))
    print_formatted_text(
        HTML(f'  <style fg="ansibrightblack">Detection method: {term_info["detection_method"]}</style>')
    )
    print()

    # Show colors for detected theme
    show_theme_colors(detected_theme)

    print()  # Final newline


if __name__ == "__main__":
    main()
