# Jag Studio on Google Cloud

One website for **Studio** (AI base photos and recolouring in batches), **Recolor** (recolour by hand), the
**Agent** ("make 1000 Men models in half sleeve and hoodie") and **Google Drive** (every finished photo uploads
itself to your Drive folder). It runs on one small Google Cloud server, so batches keep going when your PC is off.

**Cost:** the server is an `e2-medium` with a 100 GB disk in Mumbai, about **US$35 (₹3,300) a month**.
Gemini images are extra, as before (about ₹4.95 per AI base photo in batch mode). Stop the server when you
don't need it (step 6) and you pay only for the disk (about ₹950 a month).

---

## 1. Open Google Cloud Shell (2 minutes)

1. Go to **console.cloud.google.com** and sign in with the Google account you use for Gemini.
2. At the top, pick the **project** your Gemini API key belongs to (AI Studio → API keys shows its name).
   Billing is already on for it, because Gemini works.
3. Click the **>_** icon (top right, "Activate Cloud Shell"). A terminal opens at the bottom of the page.

## 2. Put Jag Studio on the server (about 10 minutes, mostly waiting)

1. In Cloud Shell, click **⋮ → Upload** and choose `jagstudio-cloud.zip`.
2. Type these lines and press Enter after each:
   ```
   unzip -o jagstudio-cloud.zip -d jagimage
   cd jagimage
   bash deploy/gcp/setup.sh
   ```
3. If Google asks to **Authorize** Cloud Shell, click Authorize. If it asks about an SSH key passphrase, just press
   Enter twice.
4. It asks you to **choose a password** for the website (8 or more characters). Keep it safe.
5. At the end it prints two lines. Keep both:
   - `Jag Studio is at: https://34-93-xx-xx.sslip.io`, which is your website.
   - `Google Drive redirect address: https://34-93-xx-xx.sslip.io/drive/callback`, which you need in step 4.

The first visit can take a minute while the HTTPS certificate is issued.

## 3. Sign in and add your Gemini key (1 minute)

1. Open your website address on your phone or PC, and sign in with your password.
2. Scroll to **Engine**, paste your Gemini API key, tick **Remember on this computer** (it is kept on your server),
   and press **Save key**.

Studio works now, and so do the Agent and Recolor (at the top).

## 4. Connect Google Drive (about 10 minutes, once)

Jag Studio needs your permission to put photos in your Drive. Google requires you to make a small
"OAuth client" for that:

1. In the Cloud console menu: **APIs & Services → OAuth consent screen** (on newer consoles: **Google Auth Platform**).
   - Choose **External**, App name **Jag Studio**, your email as support and developer contact, then **Save**.
   - Under **Audience** (or **Publishing status**), press **Publish app** so it is **In production**.
     Jag Studio only asks to see the files it makes itself, so Google doesn't need to review it, and the Drive
     connection doesn't expire after 7 days as it would in testing mode.
2. **APIs & Services → Credentials → Create credentials → OAuth client ID**:
   - Application type **Web application**, name **Jag Studio**.
   - Under **Authorized redirect URIs**, add the **Google Drive redirect address** from step 2.
   - Press **Create**, and keep the **Client ID** and **Client secret** that appear.
3. Back in Cloud Shell (in the `jagimage` folder), type:
   ```
   bash deploy/gcp/drive.sh
   ```
   Paste the Client ID, then the Client secret, and press Enter for the folder name (**kinsey models 2026**).
   It updates the server.
4. On the website, the **Google Drive** panel now has **Connect Google Drive**. Press it, choose your Google account,
   and allow access. If Google shows "Google hasn't verified this app", press **Advanced → Go to Jag Studio**: it is
   your own app.

From now on, every finished photo uploads by itself, within a minute, to
**My Drive → kinsey models 2026 → (batch) → (Top__Bottom colour) → m0002_front.jpg**.

**Drive space:** the free 15 GB fills in about a day at 1,000 models a day (about 4.5 GB of photos per 1,000 models,
per style). A Google One plan (2 TB) is needed for that volume.

## 5. Use the Agent

In **Agent**, write what you want, for example:

> 1000 Men models in half sleeve and hoodie, all Pure Cotton colours, front and back

Press **Plan it**. It shows each batch, the number of photos and the AI cost. Nothing is made until you press
**Start**. Batches then run one after another, photos are checked (wrong colours, wrong background, collages
and washed-out garments are made again), recoloured and uploaded to Drive. The Studio panels below show the progress
and the real cost.

## 6. Day to day

- **Update Jag Studio** (when you get a new `jagstudio-cloud.zip`): upload it in Cloud Shell, then
  `unzip -o jagstudio-cloud.zip -d jagimage && cd jagimage && bash deploy/gcp/setup.sh`.
  Your photos, faces, keys and Drive connection stay.
- **Stop the server** when you won't use it for a while (stops the hourly charge, keeps everything):
  `gcloud compute instances stop jag-studio --zone asia-south1-a`
- **Start it again:** `gcloud compute instances start jag-studio --zone asia-south1-a`
  (same address; give it a minute).
- **Change the password:** `rm ~/.jagstudio-password && bash deploy/gcp/setup.sh`.
- **Server log** if something looks wrong:
  `gcloud compute ssh jag-studio --zone asia-south1-a --command "sudo journalctl -u jagstudio -n 100"`

## What runs where

| Piece | Where |
|---|---|
| Website, batches, recolouring | `jag-studio` server (Compute Engine, `asia-south1-a`), service `jagstudio` |
| HTTPS | Caddy on the same server, certificate from Let's Encrypt for `<address>.sslip.io` |
| Photos and faces | `/var/lib/jagstudio/studio` on the server, copied to your Drive |
| Password, Gemini key, Drive connection | `/etc/jagstudio.env` and `/var/lib/jagstudio/data` on the server (only the service can read them) |
