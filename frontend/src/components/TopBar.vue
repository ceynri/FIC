<template>
  <nav class="top_bar">
    <router-link class="logo_wrapper" to="/">
      <IconBase class="logo" icon-name="logo" width="32px" height="32px">
        <Favicon />
      </IconBase>
      <span class="name">FIC</span>
    </router-link>
    <div class="nav_wrapper">
      <router-link to="demo" class="link">Demo</router-link>
      <router-link to="compress" class="link">Compress</router-link>
      <router-link to="decompress" class="link">Decompress</router-link>
      <router-link to="about" class="link">About</router-link>
      <a class="switch_theme_btn" href="http://github.com/ceynri/FIC">
        <IconBase class="github_icon" icon-name="github" width="28" height="28">
          <GithubIcon />
        </IconBase>
      </a>
    </div>
  </nav>
</template>

<script>
import Favicon from '@/components/icons/Favicon.vue';
import GithubIcon from '@/components/icons/GithubIcon.vue';

export default {
  components: {
    Favicon,
    GithubIcon,
  },
};
</script>

<style lang="scss" scoped>
$topBarHeight: 80px;

.top_bar {
  height: $topBarHeight;
  display: flex;
  justify-content: space-between;

  box-shadow: 0 6px 64px -6px var(--shadow);

  .logo_wrapper,
  .nav_wrapper {
    display: inline-flex;
  }

  .logo_wrapper {
    color: var(--primary);
    align-items: center;
    padding: 0 40px;

    transition: filter var(--duration);

    &:hover {
      filter: contrast(150%);
    }

    .logo {
      margin-right: 16px;
    }

    .name {
      font-size: 32px;
      font-weight: bold;
    }
  }

  .link {
    color: var(--text);
    padding: 0 25px;
    display: inline-flex;
    font-size: 16px;
    justify-content: center;
    align-items: center;
    transition: background-color var(--duration);

    &:hover {
      background-color: var(--secondary);
      color: var(--bg2);
    }
  }

  .switch_theme_btn {
    @extend .link;

    width: $topBarHeight;
    padding: 0;

    .github_icon {
      margin-bottom: 0.2em;
    }
  }
}
</style>
