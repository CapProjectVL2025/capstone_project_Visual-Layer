### Access your `ssh_config` file

Note: The following instructions are written with the assumption that you are using VS Code.

1. Save the private key sent to you via email in your `.ssh` directory with a filename of your choice OTHER than `id_rsa`. DO NOT SHARE THIS WITH ANYONE.

2. Click on the "Open Remote Window" button in the bottom right corner. This will open a dialog at the top of your window.

3. Within the resultant dialog, click on "Connect to Host..." and then click on "Configure SSH Hosts..."

4. Select the config file within the `.ssh` directory. (Can identify this by finding the endpoint that is preceded by `.ssh` in the file path)

5. Copy-paste the below snippet into the top of the config file. Replace `<path/to/ssh_key>` with the path to the SSH private key file. Save.

```
Host capstone
  HostName 20.118.32.154
  User azureuser
  IdentityFile <path/to/ssh_key>
  IdentitiesOnly yes
```

6. Click on the "Open Remote Window" button in the bottom right corner. This will open a dialog at the top of your window.

7. Within the resultant dialog, click on "Connect to Host..." and then click on "capstone"

Aside from some following questions that you may have to answer, there you go! You are now connected to the Azure VM!