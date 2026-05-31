"""
Script to create 5 test users for the similarity search system.
"""

import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoProject.settings')
django.setup()

from base.models import User

# Define 5 test users
test_users = [
    {'username': 'alice', 'password': 'password123', 'email': 'alice@example.com'},
    {'username': 'bob', 'password': 'password123', 'email': 'bob@example.com'},
    {'username': 'charlie', 'password': 'password123', 'email': 'charlie@example.com'},
    {'username': 'diana', 'password': 'password123', 'email': 'diana@example.com'},
    {'username': 'eve', 'password': 'password123', 'email': 'eve@example.com'},
]

print("Creating test users...")
for user_data in test_users:
    # Check if user already exists
    if User.objects.filter(username=user_data['username']).exists():
        print(f"✓ User '{user_data['username']}' already exists")
    else:
        user = User(username=user_data['username'], email=user_data['email'])
        user.set_password(user_data['password'])
        user.save()
        print(f"✓ Created user '{user_data['username']}'")

print("\nTest users created successfully!")
print("\nYou can login with any of these credentials:")
print("Usernames: alice, bob, charlie, diana, eve")
print("Password: password123")

