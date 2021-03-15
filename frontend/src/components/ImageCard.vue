<template>
  <div class="card shadow">
    <img
      class="image"
      :class="{ reverse: topInfo, icon: isIcon }"
      :src="src"
      :alt="name"
      ref="image"
    />
    <div class="image_info" :class="{ reverse: topInfo }">
      <div class="image_name">{{ name }}</div>
      <div class="image_ext_info"><slot /></div>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    name: String,
    src: String,
    topInfo: {
      type: Boolean,
      default: false,
    },
    isIcon: {
      type: Boolean,
      default: false,
    },
  },
};
</script>

<style lang="scss" scoped>
.card {
  $padding: 12px;

  display: flex;
  flex-direction: column;
  padding: $padding;

  .image {
    &.reverse {
      order: 1;
    }

    max-width: 100%;
    max-height: 100%;
  }

  .image_info {
    $margin: 12px;

    color: var(--text2);
    margin: $margin 6px 0;

    &.reverse {
      order: 0;
      margin: 0 6px $margin;
    }

    .image_name {
      @include no-wrap;
      color: var(--text);
      font-size: 16px;
    }

    .image_ext_info {
      & > * {
        font-size: 12px;
        margin-top: 8px;
      }
    }
  }
}
</style>
