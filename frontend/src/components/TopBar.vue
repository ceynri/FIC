<template>
  <nav class="top_bar">
    <router-link class="logo_wrapper" to="/">
      <IconBase
        class="logo"
        icon-name="logo"
        width="32px"
        height="32px"
        :icon-color="cssVars.primary"
      >
        <Favicon />
      </IconBase>
      <span class="name">FIC</span>
    </router-link>
    <div class="nav_wrapper">
      <router-link to="compression" class="link">Compress</router-link>
      <router-link to="decompression" class="link">Decompress</router-link>
      <router-link to="about" class="link">About</router-link>
      <a class="link" href="http://github.com/ceynri/FIC">Github</a>
      <button class="switchTheme" @mouseover="isHover = true" @mouseleave="isHover = false">
        <IconBase name="dark theme" width="20" height="20" :icon-color="iconColor">
          <DarkThemeIcon />
        </IconBase>
      </button>
    </div>
  </nav>
</template>

<script>
import Favicon from '@/components/icons/Favicon.vue';
import DarkThemeIcon from '@/components/icons/DarkThemeIcon.vue';

export default {
  data() {
    return {
      isHover: false,
    };
  },
  computed: {
    iconColor() {
      return this.isHover ? this.cssVars.bg : this.cssVars.secondary;
    },
  },
  components: {
    Favicon,
    DarkThemeIcon,
  },
};
</script>

<style lang="scss" scoped>
$topBarHeight: 80px;

.top_bar {
  height: $topBarHeight;
  display: flex;
  justify-content: space-between;

  box-shadow: #2c2c2c22 0 0 64px;

  .logo_wrapper,
  .nav_wrapper {
    display: inline-flex;
  }

  .logo_wrapper {
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
      color: var(--primary);
      margin-top: 8px;
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
      color: #fff;
    }
  }

  .switchTheme {
    @extend .link;
    border: none;
    background-color: #fff;
    width: $topBarHeight;
    padding: 0 0 4px;
    transition: all var(--duration);

    &:hover {
      background-color: var(--secondary);
    }
  }
}
</style>
