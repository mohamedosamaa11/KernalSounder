"""
KernalSounder — refusal rules for dangerous requests.

Runs on the normalized transcript before intent matching. Anything that
targets the operating system, disks, many files at once, admin rights,
power state, installers, raw shell commands or secrets is refused with a
specific, spoken reason, even when no action could have run anyway.
This is defense in depth: the validator and the workspace sandbox still
apply to everything that passes.
"""

from __future__ import annotations

import re
from typing import Optional

_DESTRUCTIVE = (
    r"delete|deleat|remove|erase|wipe|destroy|format|kill|close|stop|terminate|uninstall|"
    r"disable|break|corrupt|clear|empty|reset|overwrite|shred|encrypt|move"
)
_PROTECTED_TARGETS = (
    r"system|system32|system files|windows|os|operating system|kernel|root|boot|bootloader|"
    r"bios|uefi|registry|drive|drives|disk|disks|hard disk|hard drive|ssd|partition|c drive|"
    r"drive c|everything|all|all files|all folders|whole|entire|computer|pc|laptop|machine|"
    r"home|home folder|users|user folder|program files|programs|antivirus|defender|firewall|"
    r"security|explorer|desktop|documents|downloads|trash|recycle bin|workspace|etc|bin|usr|var"
)

_RULES = (
    (re.compile(rf"\b(?:{_DESTRUCTIVE})\b(?:\s+[\w']+){{0,3}}?\s+(?:the\s+|my\s+|this\s+)?"
                rf"(?P<target>{_PROTECTED_TARGETS})(?![\w.])"),
     "I won't touch {target}. I can only delete single files inside the KernalSounder workspace, "
     "and only after you confirm."),
    (re.compile(r"\b(?:shut ?down|power ?off|turn off (?:the )?(?:computer|pc|laptop|machine)|"
                r"restart (?:the )?(?:computer|pc|laptop|system|windows|machine)|reboot|log ?off|"
                r"log ?out|sign ?out|hibernate|lock (?:the )?(?:screen|computer|pc))\b"),
     "I can't shut down, restart or log off the computer."),
    (re.compile(r"\b(?:sudo|su|admin|administrator|as root|root access|elevated?|elevation|"
                r"privileges?|run as)\b"),
     "I never run anything as administrator or root."),
    (re.compile(r"\b(?:format|diskpart|fdisk|mkfs|partition)\b"),
     "I can't format or partition disks."),
    (re.compile(r"\b(?:registry|regedit|group policy|gpedit|bios|uefi|boot ?loader|grub|"
                r"msconfig|services)\b"),
     "System configuration is off limits by voice."),
    (re.compile(r"\b(?:install|uninstall|reinstall|winget|choco|apt|apt-get|pip|npm|snap install)\b"),
     "Installing or uninstalling software is not allowed by voice."),
    (re.compile(r"(?:^|\s)(?:rm|rmdir|del|rd|chmod|chown|killall|pkill|taskkill|curl|wget|"
                r"execute|exec|eval)(?:\s|$)|\brun (?:a |the |this )?(?:command|script|code|program)\b|"
                r"\b(?:shell|terminal|cmd|powershell|bash) command\b|rm -rf|--no-preserve-root"),
     "I don't run raw commands. Ask me for one of my actions instead."),
    (re.compile(r"\b(?:passwords?|credentials?|ssh keys?|private keys?|api keys?|tokens?|"
                r"bank|credit cards?)\b"),
     "I don't handle passwords, keys or financial data."),
)


def danger_reason(text: str) -> Optional[str]:
    """Return a refusal sentence when the request is dangerous, else None."""
    lowered = f" {(text or '').lower()} "
    for pattern, message in _RULES:
        match = pattern.search(lowered)
        if match:
            target = match.groupdict().get("target") if match.groupdict() else None
            return message.format(target=f"'{target}'" if target else "that")
    return None
